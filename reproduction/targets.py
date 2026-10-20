"""A panel of two-state candidates, scored and ranked into the table that is the deliverable.

Scores a list at one stated collection plan so the rows are comparable: no per-target fitting enters.
Two counts per row: `n_floor` from the model-only rungs assumes the pose known and is a FLOOR;
`n_marginal` from `chain` costs a simulated pool and is the number to quote. A pose cost does not
transfer between targets, so a row with no measured rung reads blank rather than borrowing one. A
rejected candidate is reported alongside the rows, not dropped.
"""

import json
import math
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from glob import glob
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import PoseGrid, pose_bank
from cryodisc.microscope import CollectionPlan
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs import local_information
from cryodisc.specimen.structures import UNSET, ChainIdKind, ConformerPair, align_conformers
from cryodisc.statistics.power import (
    information_per_particle,
    n_for_error,
    n_for_error_from_information,
    saturation,
)

_ATOMIC_MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 12: 24.305, 15: 30.974, 16: 32.06, 26: 55.845}
"""Standard atomic weights of the elements a protein assembly is made of, in daltons.

Enough for a mass to a percent or so, which is all a table column needs; anything not listed falls
back to roughly twice its atomic number, which is the trend across this range.
"""

_FLOOR_TOLERANCE = 1.02
"""How close to the binomial counting floor a count must sit to be marked as quoting it back.

A rounding allowance: two percent lets a row that is on the floor in every meaningful sense still
say so, without admitting rows that are not.
"""

_MAX_PLAUSIBLE_RMSD = 15.0
"""C-alpha RMSD in A above which a pair is CHECKED, not rejected.

A change of tens of angstroms is often not a conformational change; it is two structures that were not
the same molecule, or the same molecule in two frames. Both are testable, so this threshold triggers
the residue-type identity test (see `_MIN_CORRESPONDENCE_IDENTITY`) rather than standing in for it --
real motions reach well past this line, so RMSD alone truncates the distribution rather than cleans it.
"""

_MIN_CORRESPONDENCE_IDENTITY = 0.90
"""Share of matched C-alpha that must carry the SAME residue type for a large motion to be believed.

What `_MAX_PLAUSIBLE_RMSD` was trying to measure. A correspondence that has slipped in numbering pairs
unrelated residues and lands near chance, while a real conformational change keeps identity near one;
0.90 sits below accepted pairs and far above a slipped one, without truncating the motion range.
"""


def _correspondence_identity(pair: ConformerPair) -> float | None:
    """Share of matched C-alpha positions whose two residues are the same type.

    A slipped correspondence -- by numbering, by chain, or by pairing two different molecules -- lines
    up unrelated residues and the share collapses; a real conformational change, however large, keeps
    it at essentially one.

    Args:
        pair: An aligned `ConformerPair`.

    Returns:
        The share in `[0, 1]`, or `None` when the test cannot be run at all -- no C-alpha, or two
        states of different length (a supported case when a ligation state differs). `None` means
        "cannot tell", and the caller refuses; answering `1.0` would ADMIT the rows this exists to catch.
    """
    state_a, state_b = pair.state_a, pair.state_b
    if len(state_a) != len(state_b):
        return None
    is_ca = (state_a["atom"].astype(str).str.strip() == "CA").to_numpy()
    if not is_ca.any():
        return None
    left = state_a.loc[is_ca, "residue"].astype(str).str.strip().to_numpy()
    right = state_b.loc[is_ca, "residue"].astype(str).str.strip().to_numpy()
    return float((left == right).mean())


@dataclass(frozen=True)
class Candidate:
    """One two-state pair, named and located but not yet read.

    Args:
        name: How the row is labeled.
        path_a: Structure of state A, in any format `gemmi` reads.
        path_b: Structure of state B, the same assembly in the other conformation.
        motion: Free text naming the kind of change, for the table.
        note: Anything else worth carrying to the row.
        assembly: Biological assembly to build, as `structures.align_conformers` takes it. Leave the
            per-entry fields unset to apply this to both.
        assembly_a: State A's own assembly, for a pair whose entries express the molecule differently.
        assembly_b: State B's own assembly.
        chains: Author chains to keep, or `None` for all of them.
        chains_a: State A's own chains.
        chains_b: State B's own chains.
        labels_a: Chain names to rename state A's assembly-generated copies onto, in `(copy, chain)`
            order. Matches a generated assembly against a deposited one.
        labels_b: The same for state B.
        ligands: Residue names of non-water HETATM to keep.
        provenance: Panel fields describing the row rather than selecting its atoms, carried
            uninterpreted to `TargetRow`. `PROVENANCE_KEYS` says which travel.

    There are no per-candidate `pixel_size` or `box_size` overrides: a per-row sampling makes that row
    incomparable with the rest of the panel.

    Rationale: docs/rationale/targets.md
    """

    name: str
    path_a: str | Path
    path_b: str | Path
    motion: str = ""
    note: str = ""
    assembly: str | None = UNSET
    assembly_a: str | None = UNSET
    assembly_b: str | None = UNSET
    chains: tuple[str, ...] | None = UNSET
    chains_a: tuple[str, ...] | None = UNSET
    chains_b: tuple[str, ...] | None = UNSET
    labels_a: tuple[str, ...] | None = None
    labels_b: tuple[str, ...] | None = None
    ligands: tuple[str, ...] = ()
    provenance: Mapping[str, Any] | None = None
    chain_id_kind: ChainIdKind = "auth_asym_id"

    def align(self, *, hydrogens: bool = False, min_shared_fraction: float | None = None) -> ConformerPair:
        """Build the pair this candidate names, as every arm of the panel must build it.

        Two arms read the same candidate and neither can carry the pair from the other, so the
        `align_conformers` call is spelled ONCE here rather than by hand in each -- otherwise a field
        added to this class reaches one arm and not the other, and the two describe different assemblies
        of one candidate. `hydrogens` is an ABLATION knob and travels the same way.

        Args:
            hydrogens: Keep modeled hydrogens. Off is the panel's normal setting; on is the ablation.
            min_shared_fraction: Overlap below which the correspondence is refused. `None` keeps
                `align_conformers`' own default.

        Returns:
            The aligned pair, from `structures.align_conformers`.

        Rationale: docs/rationale/targets.md
        """
        return align_conformers(
            self.path_a,
            self.path_b,
            assembly=self.assembly,
            assembly_a=self.assembly_a,
            assembly_b=self.assembly_b,
            chains=self.chains,
            chains_a=self.chains_a,
            chains_b=self.chains_b,
            labels_a=self.labels_a,
            labels_b=self.labels_b,
            ligands=self.ligands,
            chain_id_kind=self.chain_id_kind,
            hydrogens=hydrogens,
            # omitted rather than passed as None, so `align_conformers` keeps its own default
            **({} if min_shared_fraction is None else {"min_shared_fraction": min_shared_fraction}),
        )


RCSB = "https://files.rcsb.org/download/{pdb_id}.{extension}"

CIF_ONLY = ("cif",)
"""Formats for a row whose chains are `label_asym_id`, which no legacy PDB file records."""

FORMATS = ("pdb", "cif")
"""Download formats in preference order.

The `cif` fallback is not a nicety: RCSB serves no legacy PDB file for a deposition that exceeds the
format's limits, which is most large recent cryo-EM work, so many newer entries 404 on `.pdb` and 200
on `.cif`. `gemmi.read_structure` dispatches on the extension and reads either.
"""


PROVENANCE_KEYS = ("accession", "canonical_mass_kda", "same_entry", "source", "rotation_deg", "motions")
"""Panel fields carried onto the scored row, describing it rather than selecting its atoms.

`accession` is load-bearing: with one-pair-per-accession deferred from selection to fitting, a fit
that does not group on it is pseudo-replicated. `canonical_mass_kda` travels for the mirror reason:
it is the mass the population was SCREENED on, while `TargetRow.mass_kda` is the mass the fit uses,
and the two diverge on a polyprotein.
"""


def _chain_id_kind_of(row: dict[str, object]) -> ChainIdKind:
    """A row's chain naming, refusing a value that is neither.

    `_read_assembly` branches on `== "label_asym_id"` with an unguarded `else`, so a typo
    (`"label_asym"`) or a `null` reads as author chains and builds a pair from the wrong subunit in
    silence -- which is exactly what `align_conformers` promises not to do. A panel is hand-editable,
    so this is reachable by typing.

    Args:
        row: One panel candidate.

    Returns:
        The naming, defaulting to `"auth_asym_id"` when the row does not say -- every hand-written
        panel names author chains, so an absent key means what it has always meant.

    Raises:
        ValueError: If the row names something else.
    """
    kind = row.get("chain_id_kind", "auth_asym_id")
    if kind not in ("auth_asym_id", "label_asym_id"):
        raise ValueError(
            f"{row.get('name', '?')}: chain_id_kind is {kind!r}, which is neither 'auth_asym_id' nor "
            "'label_asym_id'. It selects which chain naming the row's chains are read in, and an "
            "unrecognized value would silently read them as author chains."
        )
    return kind


def fetch_structure(pdb_id: str, data_dir: Path, *, fetch: bool, formats: tuple[str, ...] = FORMATS) -> Path:
    """The local path of one deposition, downloading it only when asked.

    Tries each of `formats` in turn and keeps the first that exists, locally or at RCSB. A `.cif`
    already on disk is preferred over downloading a `.pdb`.

    Args:
        pdb_id: Four-character identifier.
        data_dir: Where structures live.
        fetch: Download when absent. Off by default, so a run with no outbound network fails naming
            the file it wanted rather than timing out inside a scoring loop.
        formats: Extensions in preference order. `CIF_ONLY` for a row naming `label_asym_id`, which
            only an mmCIF carries; the default prefers `.pdb`.

    Returns:
        The path, whose extension says which format was obtained.

    Raises:
        FileNotFoundError: If nothing is on disk and `fetch` is off, naming every path tried.
        urllib.error.HTTPError: If every format 404s, re-raising the LAST failure.
    """
    for extension in formats:
        path = data_dir / f"{pdb_id}.{extension}"
        if path.exists():
            return path
    if not fetch:
        tried = ", ".join(str(data_dir / f"{pdb_id}.{extension}") for extension in formats)
        raise FileNotFoundError(f"none of {tried} exists; pass --fetch to download from RCSB")

    data_dir.mkdir(parents=True, exist_ok=True)
    failure: urllib.error.HTTPError | None = None
    for extension in formats:
        path = data_dir / f"{pdb_id}.{extension}"
        try:
            with urllib.request.urlopen(RCSB.format(pdb_id=pdb_id, extension=extension)) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            failure = error
            continue
        path.write_bytes(body)
        return path
    assert failure is not None
    raise failure


def _assembly_of(row: dict, key: str = "assembly") -> str | None:
    """The biological assembly a panel row asks for, as `align_conformers` names them.

    Args:
        row: One candidate from the panel file.
        key: Which field to read -- `assembly` for both entries, or `assembly_a`/`assembly_b` for one
            each, which is what a pair needs when one entry deposits the assembly and the other
            generates it.

    Returns:
        The id as a STRING, `UNSET` when the row is silent about this field, and `None` when it says so
        explicitly -- the two are different, `null` meaning the deposited coordinates are already the
        whole molecule. A deposition names its assemblies `'1'`, `'2'`, so an integer here fails with a
        message that reads like a missing assembly rather than like a type.
    """
    if key not in row:
        # UNSET and not "1": a row naming only `assembly_a` must leave the shared field unset, or
        # `align_conformers` refuses the pair for passing both a shared value and a per-entry one
        return UNSET
    value = row[key]
    return None if value is None else str(value)


def _chains_of(row: dict, key: str) -> tuple[str, ...] | None:
    """One chain selection from a panel row, distinguishing absent from empty.

    Args:
        row: One candidate from the panel file.
        key: `chains`, `chains_a` or `chains_b`.

    Returns:
        The chains as a tuple, or `UNSET` where the row does not name that field.
    """
    return tuple(row[key]) if row.get(key) else UNSET


def load_panel(
    panel_path: Path, data_dir: Path, *, fetch: bool, only: Iterable[str] | None = None
) -> tuple[list[Candidate], list[str]]:
    """Read the panel description and resolve every deposition to a local file.

    Args:
        panel_path: The JSON panel. Keys beginning with an underscore are prose, so the reasoning
            that belongs with the data can live beside it, and so does any key ending in `_why`.
            A row's own `ligands` overrides the file's shared list rather than extending it.
        data_dir: Where structures live.
        fetch: Download missing depositions.
        only: Row names to keep, applied BEFORE any deposition is resolved, so a one-row-per-job panel
            cannot fail on another row's deposition.

    Returns:
        `(candidates, unresolved)` -- the candidates in the order the file lists them, and one message
        per row whose depositions could not be resolved. A row RCSB will not serve (`HTTPError`/
        `URLError`) is skipped and reported; a file absent from disk with `--fetch` off is an operator
        error, so `FileNotFoundError` propagates.

    Raises:
        ValueError: If `only` names a row the panel does not contain, listing the rows it does.
    """
    spec = json.loads(Path(panel_path).read_text())
    shared_ligands = tuple(spec.get("ligands", ()))
    rows = spec["candidates"]
    if only is not None:
        wanted = set(only)
        missing = wanted - {row["name"] for row in rows}
        if missing:
            raise ValueError(
                f"--only names {sorted(missing)}, which this panel does not contain. Its rows are "
                f"{sorted(row['name'] for row in rows)}"
            )
        rows = [row for row in rows if row["name"] in wanted]
    resolved, unresolved = [], []
    for row in rows:
        # VALIDATED here, before it picks a download format. Reading the raw field meant a typo
        # (`"label_asym"`) chose `FORMATS` and, under `--fetch`, downloaded a `.pdb` for a label row
        # before the refusal below ever ran -- and on a row whose depositions 404, the typo was
        # swallowed into `unresolved` and never reported at all.
        kind = _chain_id_kind_of(row)
        # a row naming `label_asym_id` needs the mmCIF: `FORMATS` prefers `.pdb`, which records no
        # label ids at all, and 50 of the 68 BioEmu entries are already on disk in that format
        formats = CIF_ONLY if kind == "label_asym_id" else FORMATS
        try:
            paths = (
                fetch_structure(row["pdb_a"], Path(data_dir), fetch=fetch, formats=formats),
                fetch_structure(row["pdb_b"], Path(data_dir), fetch=fetch, formats=formats),
            )
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            # NOT `FileNotFoundError`, which stays fatal on purpose: with `--fetch` off, an absent
            # file means the operator must hear "pass --fetch" naming the path rather than lose the row
            # silently -- especially a `label_asym_id` row, which demands the mmCIF many BioEmu entries
            # are on disk only as `.pdb`.
            unresolved.append(f"{row['name']}: could not resolve {row['pdb_a']}/{row['pdb_b']} -- {error}")
            continue
        resolved.append((row, paths))

    candidates = [
        Candidate(
            name=row["name"],
            path_a=path_a,
            path_b=path_b,
            motion=row.get("motion", ""),
            note=row.get("note", ""),
            chains=_chains_of(row, "chains"),
            chains_a=_chains_of(row, "chains_a"),
            chains_b=_chains_of(row, "chains_b"),
            # the copy-to-chain map, which is what lets a GENERATED assembly be matched against a
            # deposited one. See `structures._relabel_copies` for why the key needs it
            labels_a=tuple(row["labels_a"]) if row.get("labels_a") else None,
            labels_b=tuple(row["labels_b"]) if row.get("labels_b") else None,
            # per row when the row says so: the allowlist expresses a LIGATION state, specific to the
            # pair, so naming ligands on a row that has none is at best noise. A shared list is still
            # right for a cofactor every row carries
            ligands=tuple(row["ligands"]) if "ligands" in row else shared_ligands,
            # assembly ids are STRINGS as the file names them, so a JSON author writing 1 gets it
            # coerced rather than a rejection reading "no assembly 1; it has ['1']". null means the
            # deposited coordinates are the whole molecule, which a file with no REMARK 350 needs
            assembly=_assembly_of(row),
            assembly_a=_assembly_of(row, "assembly_a"),
            assembly_b=_assembly_of(row, "assembly_b"),
            # Which naming this row's chains are written in. Defaulted because every hand-written panel
            # names author chains; only the generated BioEmu rows do not.
            chain_id_kind=_chain_id_kind_of(row),  # validated above, before the format was chosen
            # only keys the row carries, so a hand-written panel gets `None` rather than a dict of
            # nulls reading as "recorded, and absent" where the truth is "never recorded"
            provenance={key: row[key] for key in PROVENANCE_KEYS if key in row} or None,
        )
        for row, (path_a, path_b) in resolved
    ]
    return candidates, unresolved


def load_panels(patterns: str | Iterable[str], data_dir: Path, *, fetch: bool) -> tuple[list[Candidate], list[str]]:
    """`load_panel` over every file several globs match, refusing a pair that appears twice.

    The refusal is the point. `data/survey_relax_*.json` re-emits every candidate of the panels it
    widened, so a glob that reaches both reads each pair twice, and an exact duplicate moves no
    residual: a fit over the doubled set converges to a plausible answer with every weight wrong.
    `preprocess.survey.SURVEY_PANELS` is the disjoint list; this is the guard for callers that widen it.

    Args:
        patterns: A glob or a sequence of them over panel files.
        data_dir: Where the depositions live, as `load_panel` takes it.
        fetch: Whether to download a missing deposition.

    Returns:
        `(candidates, unresolved)`, concatenated in sorted path order.

    Raises:
        SystemExit: If the globs match no panel, or if one pair -- name AND chains -- appears
            in two of them.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    paths = sorted({path for pattern in patterns for path in glob(pattern)})
    if not paths:
        raise SystemExit(f"{list(patterns)} matched no panel")
    candidates: list[Candidate] = []
    unresolved: list[str] = []
    origin: dict[tuple, str] = {}
    for path in paths:
        found, missing = load_panel(Path(path), data_dir=data_dir, fetch=fetch)
        unresolved.extend(missing)
        for candidate in found:
            # Keyed on the chains as well as the name: a name is a display string that omits them,
            # unique only while one pair per accession is kept. Scoring every chain pair makes some
            # names cover two distinct chain pairs -- two pairs, not one pair twice.
            key = (candidate.name, tuple(candidate.chains_a or ()), tuple(candidate.chains_b or ()))
            if key in origin:
                raise SystemExit(
                    f"{candidate.name} (chains {candidate.chains_a}/{candidate.chains_b}) appears in "
                    f"both {origin[key]} and {path}; check the panel globs"
                )
            origin[key] = path
            candidates.append(candidate)
    return candidates, unresolved


class CandidateRejected(Exception):
    """A candidate that cannot be scored, carrying the reason so it is dropped loudly."""


@dataclass(frozen=True)
class TargetRow:
    """One scored candidate.

    Args:
        name: The candidate's name.
        motion: Its motion description.
        mass_kda: Mass of the matched atom set, in kDa.
        ca_rmsd: C-alpha RMSD after superposition, in A.
        radius: Distance from the center to the furthest atom, in A.
        box_size: Box the templates were rendered at.
        d2_known_pose: `known_pose` averaged over the pose grid -- the oracle, and the CEILING of the
            ladder (the LARGEST rung, bounding every other). Records carry this under `d2_floor` too;
            see `as_dict`.
        d2_crlb: `rungs.local.LocalRung.crlb` averaged over the same grid -- the appendix's `d^2_CRLB`
            (A9)-(A11), the Schur complement of the pose block at the state separation, the rung
            `chain.ChainResult` calls `local_pose`. Says how much of `d2_known_pose` survives an
            infinitesimal pose confusion.
        information: `I_p` at `d2_known_pose`.
        saturation: How close that sits to the perfect-label ceiling. Above 0.99 the count is set by
            the ceiling rather than by `d2`.
        n_floor: Particles for the target error at `d2_known_pose`. A FLOOR, and on a small target an
            order of magnitude below the truth.
        n_marginal: Particles at a measured marginal rung, or `None` when none was run.
        marginal_extrapolated: Whether that rung was extrapolated over grids or read off one, in which
            case it is a lower bound on the rung and so an upper bound on the count.
        pose_efficiency: Measured `d2_marginal / d2_known_pose` for this pair, or `None`. Not
            transferable between pairs -- see the module docstring.
        note: The candidate's note.
        n_matched: Matched C-alpha the superposition used, the input to the >= 20 guard.
        identity: Share of those carrying the same residue type -- the `_MIN_CORRESPONDENCE_IDENTITY`
            input -- or `None` where it cannot be computed.
        shared_fraction: Share of the larger atom list the two entries had in common. All three are
            recorded and not only acted on, so a threshold stays revisitable on the scored
            population rather than deciding which rows exist.
        provenance: The candidate's `provenance`, uninterpreted. See `PROVENANCE_KEYS`.
    """

    name: str
    motion: str
    mass_kda: float
    ca_rmsd: float
    radius: float
    box_size: int
    d2_known_pose: float
    d2_crlb: float
    information: float
    saturation: float
    n_floor: float
    n_marginal: float | None
    marginal_extrapolated: bool
    pose_efficiency: float | None
    note: str
    d2_marginal: float | None = None
    information_marginal: float | None = None
    n_matched: int | None = None
    identity: float | None = None
    shared_fraction: float | None = None
    provenance: Mapping[str, Any] | None = None

    def as_dict(self) -> dict:
        """The row as JSON, carrying `d2_known_pose` under its old name as well.

        A method rather than a bare `asdict` at each driver, so the next thing that writes rows finds
        the serializer on the class -- reaching for `dataclasses.asdict` would silently drop the alias.

        `d2_floor` is written permanently: it is the key in the tracked `results/` files that are the
        provenance for published numbers, and consumers index it by name (including
        `reproduction/experiments/figures/marginalization_floor.py`, which filters rows on its PRESENCE). Writing
        both names is the only form of this alias that is correct on old and new files at once.

        Returns:
            Every field by its current name, plus `d2_floor` mirroring `d2_known_pose`.
        """
        record = asdict(self)
        return record | {"d2_floor": record["d2_known_pose"]}


def _molecular_mass(pair: ConformerPair) -> float:
    """Mass of state A's atom set in kDa.

    Args:
        pair: The aligned pair. State A is read, which is its corresponding atoms plus its own
            allowlisted ligands -- so on a pair whose two states bind different ligands the mass is
            state A's and the two differ, by `alignment.n_state_specific` atoms' worth. That is tens
            of daltons against tens of kDa, and reporting one state's is right for a column whose
            purpose is the mass band a target sits in.

    Returns:
        The mass in kDa, from standard atomic weights.
    """
    atomic_number = pair.state_a["atomic_number"].to_numpy()
    weights = np.array([_ATOMIC_MASS.get(int(z), 2.0 * int(z)) for z in atomic_number])
    return float(weights.sum() / 1000.0)


def score_target(
    candidate: Candidate,
    plan: CollectionPlan,
    *,
    pose_grid: PoseGrid | None = None,
    rmsf: float = 0.5,
    error_target: float = 0.05,
    fraction: float = 0.5,
    box_size: int = 128,
    hydrogens: bool = False,
    form_factor: str = "peng",
    min_shared_fraction: float | None = None,
    max_plausible_rmsd: float | None = _MAX_PLAUSIBLE_RMSD,
) -> TargetRow:
    """Read one candidate and score its model-only rungs.

    This is the cheap arm: no pool is simulated and no fit is run, so every number here assumes the
    pose known. `n_marginal` is left `None` for `attach_marginal` to fill from a `chain` run.

    Args:
        candidate: The pair to score.
        plan: The collection, shared across the panel so the rows are comparable.
        pose_grid: Bank the rungs are averaged over. `None` gives a coarse default.
        rmsf: Per-axis intra-state RMS fluctuation in A.
        error_target: Fractional error the particle counts are quoted at.
        fraction: Population fraction of state A the counts are quoted at, shared across the panel with
            `attach_marginal`'s, which checks it.
        box_size: Image size in pixels, shared by every row so the rows stay comparable.
        hydrogens: Keep modeled hydrogens; the E7 ablation. See `Candidate.align`.
        form_factor: `"peng"`, or `"z"` for E7's form-factor ablation. See
            `forward.cryojax_model._to_atomic_number_weighting`.
        min_shared_fraction: Overlap below which the correspondence is refused; `None` keeps
            `align_conformers`' default. Lowering it SCORES a partial correspondence rather than
            discarding it, and `TargetRow.shared_fraction` records what the row actually had -- a
            d^2 over 60% of a molecule is not comparable to one over 99%, so the row must be filtered
            on that field rather than trusted because it exists.
        max_plausible_rmsd: Ca RMSD in A above which a pair whose residue-type identity CANNOT be
            computed is refused. `None` disables it, the honest setting: RMSD does not separate a large
            motion from a wrong correspondence, and `identity` is the test it was standing in for.

    Returns:
        The row.

    Raises:
        CandidateRejected: If the pair cannot be read or fails a correspondence guard. The underlying
            error is carried in the message rather than propagating, so one bad row cannot take a panel
            down.

    Rationale: docs/rationale/targets.md
    """
    try:
        pair = candidate.align(hydrogens=hydrogens, min_shared_fraction=min_shared_fraction)
    except Exception as error:
        raise CandidateRejected(f"{candidate.name}: could not align, {error}") from error

    # Unconditional: RMSD does not separate a large motion from a wrong correspondence in either
    # direction, so gating this check on RMSD would leave a slipped correspondence at a modest RMSD
    # unchecked. Identity is cheap (one pass over matched C-alpha), so there is nothing to buy by
    # asking conditionally.
    identity = _correspondence_identity(pair)
    if identity is None:
        # refuse only when the RMSD also says something is wrong: unequal atom counts are ordinary for
        # a pair that differs in modeled residues, but above the threshold the fact is unrecoverable
        if max_plausible_rmsd is not None and pair.alignment.rmsd > max_plausible_rmsd:
            raise CandidateRejected(
                f"{candidate.name}: C-alpha RMSD {pair.alignment.rmsd:.1f} A exceeds "
                f"{max_plausible_rmsd:.0f} A and the correspondence cannot be checked, the two "
                f"states having {len(pair.state_a)} and {len(pair.state_b)} atoms"
            )
    elif identity < _MIN_CORRESPONDENCE_IDENTITY:
        raise CandidateRejected(
            f"{candidate.name}: only {identity:.0%} of matched C-alpha carry the same residue type "
            f"(C-alpha RMSD {pair.alignment.rmsd:.1f} A), so the correspondence is wrong rather than "
            "the motion large"
        )

    grid = pose_bank(96) if pose_grid is None else pose_grid
    minimum = pair.alignment.minimum_box_size(plan.pixel_size)
    if box_size < minimum:
        raise CandidateRejected(
            f"{candidate.name}: box {box_size} is below the {minimum} its "
            f"{pair.alignment.radius:.0f} A radius needs, so the templates would be clipped"
        )

    model_a, model_b = pair.models(plan, box_size=box_size, form_factor=form_factor)
    spectrum = noise_spectrum(plan, box_size, rmsf=rmsf)
    # the same accessor `chain.run_chain` evaluates its model-only rungs through, since
    # `attach_marginal` compares the two; see `CollectionPlan.mid_defocus`
    optics = CTFParams.from_plan(plan, plan.mid_defocus)
    local = local_information(model_a, model_b, grid.angles, spectrum, ctf_params=optics)

    d2_known_pose = float(jnp.mean(local.known_pose))
    information = float(information_per_particle(jnp.asarray(d2_known_pose), fraction=fraction))
    return TargetRow(
        name=candidate.name,
        motion=candidate.motion,
        mass_kda=_molecular_mass(pair),
        ca_rmsd=float(pair.alignment.rmsd),
        radius=float(pair.alignment.radius),
        box_size=box_size,
        d2_known_pose=d2_known_pose,
        # `local_information` returns both rungs from one call, so the CRLB rung is already paid for
        d2_crlb=float(jnp.mean(local.crlb)),
        information=information,
        saturation=float(saturation(jnp.asarray(information), fraction=fraction)),
        n_floor=float(n_for_error(jnp.asarray(d2_known_pose), error_target=error_target, fraction=fraction)),
        n_marginal=None,
        marginal_extrapolated=False,
        pose_efficiency=None,
        note=candidate.note,
        n_matched=int(pair.alignment.n_matched),
        shared_fraction=float(pair.alignment.shared_fraction),
        # computed above for the guard, recorded so a threshold stays revisitable on scored rows
        identity=identity,
        provenance=candidate.provenance,
    )


def attach_marginal(
    row: TargetRow,
    d2_marginal: float,
    *,
    d2_marginal_err: float = 0.0,
    extrapolated: bool,
    error_target: float = 0.05,
    fraction: float = 0.5,
    information: float | None = None,
) -> TargetRow:
    """Put a measured marginal rung onto a row, and the count that follows from it.

    Args:
        row: The row from `score_target`.
        d2_marginal: The marginal rung for this pair, from `chain.run_chain`. Use
            `ChainResult.marginal_limit.limit` when it was extrapolated and
            `ChainResult.marginal.d2` when it was not.
        d2_marginal_err: Standard error of `d2_marginal` (`Rung.d2_err`, or `MarginalLimit.limit_err`
            when extrapolated). Zero is right only for a rung computed from the model alone; withholding
            it makes the oracle check an exact inequality on a quantity that is not exact.
        extrapolated: Whether it came from `statistics.convergence` over several grids. A single grid
            reads the rung low, so the count that follows is an UPPER bound and the table marks it.
        error_target: Fractional error the count is quoted at. Must match `score_target`'s.
        fraction: Population fraction. Must match `score_target`'s.
        information: The pool's Fisher information `I(p)`, `ChainResult.marginal.information`. When
            given, `n_marginal` is `1 / (N I(p) sigma^2)` -- the count the efficient estimator attains.
            Left `None` for a model-only rung with no pool, which falls back to the Gaussian
            `n_for_error(d2_marginal)`.

    Returns:
        The row with `n_marginal`, `marginal_extrapolated` and `pose_efficiency` filled.
        `pose_efficiency` reads slightly above one for a rung sitting at the oracle -- the scatter
        showing, which clamping would hide.

    Raises:
        ValueError: If the rung is not finite or is negative, if its error is negative, or if it
            exceeds this pair's own oracle by more than 3 sigma of that error.

    Rationale: docs/rationale/targets.md
    """
    # the deflection is non-negative by construction; zero is a valid indistinguishable reading whose
    # `n_marginal` is infinite, while a negative value cannot arise from the estimator and is refused
    if not math.isfinite(d2_marginal) or d2_marginal < 0.0:
        raise ValueError(f"the marginal rung must be finite and non-negative, got {d2_marginal}")
    if not math.isfinite(d2_marginal_err) or d2_marginal_err < 0.0:
        raise ValueError(f"the marginal rung's error must be finite and non-negative, got {d2_marginal_err}")
    if d2_marginal > row.d2_known_pose + 3.0 * d2_marginal_err:
        raise ValueError(
            f"the marginal rung {d2_marginal:.4g} +- {d2_marginal_err:.3g} exceeds {row.name}'s "
            f"known-pose oracle {row.d2_known_pose:.4g} by more than 3 sigma. "
            "Marginalizing a pose cannot beat knowing it, and the pool's own scatter does not cover "
            "the gap, so the two sides were built against different models -- a different exposure, "
            "box or pose grid. See `operating_point.OperatingPoint`."
        )
    # The count comes from the pool's Fisher information I(p) when supplied, else the Gaussian
    # `n_for_error(d2_marginal)`. `d2_marginal` is the deflection and I(p) is its Gaussian Fisher map,
    # so the two paths agree; an extrapolated rung has no pool and takes the fallback.
    n_marginal = (
        float(n_for_error_from_information(jnp.asarray(information), error_target=error_target, fraction=fraction))
        if information is not None
        else float(n_for_error(jnp.asarray(d2_marginal), error_target=error_target, fraction=fraction))
    )
    return replace(
        row,
        n_marginal=n_marginal,
        marginal_extrapolated=extrapolated,
        pose_efficiency=d2_marginal / row.d2_known_pose,
        d2_marginal=d2_marginal,
        information_marginal=information,
    )


def quotable_count(row: TargetRow) -> tuple[float, str]:
    """The particle count to actually quote for one row, and where it came from.

    Printing counts of different provenance in one column without distinguishing them is what this
    exists to prevent.

    Args:
        row: The row.

    Returns:
        `(n, source)` with `source` one of `"measured"`, `"single_grid"` or `"unmeasured"`. A row
        with no measured rung returns an infinite count labeled `"unmeasured"`, which the table
        prints blank -- see the module docstring on why there is no borrowing path.
    """
    if row.n_marginal is not None:
        return row.n_marginal, "measured" if row.marginal_extrapolated else "single_grid"
    return math.inf, "unmeasured"


def verdict(row: TargetRow, *, budget: int = 100_000) -> str:
    """A call on one row, read against the quotable count rather than the floor.

    Calling a target feasible on its floor would pass essentially every panel, the floor ignoring the
    pose cost that dominates at small mass.

    Args:
        row: The row.
        budget: Particles that count as achievable.

    Returns:
        `"clear"`, `"marginal"`, `"out of reach"`, or `"unmeasured"`. The last is not a claim about
        the target -- it is a row nobody measured, and reporting it as "out of reach" would be one.
    """
    count, source = quotable_count(row)
    if source == "unmeasured":
        return "unmeasured"
    if count <= 0.1 * budget:
        return "clear"
    if count <= budget:
        return "marginal"
    return "out of reach"


def rank_targets(
    candidates: Sequence[Candidate],
    plan: CollectionPlan,
    *,
    pose_grid: PoseGrid | None = None,
    rmsf: float = 0.5,
    error_target: float = 0.05,
    fraction: float = 0.5,
    box_size: int = 128,
    hydrogens: bool = False,
    form_factor: str = "peng",
    min_shared_fraction: float | None = None,
    max_plausible_rmsd: float | None = _MAX_PLAUSIBLE_RMSD,
) -> tuple[list[TargetRow], list[str]]:
    """Score a panel and sort it by the floor count.

    The scoring arguments are `score_target`'s, RESTATED rather than forwarded as `**kwargs` so a
    misspelled one is caught against the signature rather than deep inside the scoring loop.

    Args:
        candidates: The panel.
        plan: The collection, shared by every row.
        pose_grid: As `score_target` takes it.
        rmsf: As `score_target` takes it.
        error_target: As `score_target` takes it.
        fraction: As `score_target` takes it.
        box_size: As `score_target` takes it.
        hydrogens: As `score_target` takes it.
        form_factor: As `score_target` takes it.
        min_shared_fraction: As `score_target` takes it.
        max_plausible_rmsd: As `score_target` takes it.

    Returns:
        `(rows, rejections)`, the rows ascending in `n_floor` and the rejections in the order the
        candidates were given.
    """
    rows, rejections = [], []
    for candidate in candidates:
        try:
            rows.append(
                score_target(
                    candidate,
                    plan,
                    pose_grid=pose_grid,
                    rmsf=rmsf,
                    error_target=error_target,
                    fraction=fraction,
                    box_size=box_size,
                    hydrogens=hydrogens,
                    form_factor=form_factor,
                    min_shared_fraction=min_shared_fraction,
                    max_plausible_rmsd=max_plausible_rmsd,
                )
            )
        except CandidateRejected as rejection:
            rejections.append(str(rejection))
    rows.sort(key=lambda row: row.n_floor)
    return rows, rejections


def format_table(rows: Sequence[TargetRow], *, budget: int = 100_000) -> str:
    """The ranked panel as a markdown table.

    Do NOT read `pose eff` as the discriminating column -- its rank order inverts between pose
    conventions. The quotable column is marked per row: a bare number is that target's own extrapolated
    marginal rung; `^` is a rung read off ONE grid (a lower bound on the rung, upper bound on the
    count); `--` is a row with no measured rung, blank rather than optimistic; `=` means the count sits
    at the binomial counting floor `p(1-p)/sigma^2`. `=` (about the marginal-rung count) and `*`
    (`saturation`, the floor rung's, describing the pair) are different claims and share no condition.

    Rationale: docs/rationale/targets.md

    Args:
        rows: The scored panel.
        budget: Particles that count as achievable.

    Returns:
        The table.
    """
    header = (
        "| system | kDa | motion | Ca RMSD (A) | d2 (known pose) | pose eff | I_p/cap | "
        "N (floor) | N (quotable) | verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        count, source = quotable_count(row)
        mark = "^" if source == "single_grid" else ""
        saturated = "*" if row.saturation > 0.99 else ""
        # The floor marker goes on the COUNT, not only on `I_p/cap`: a reader comparing rows reads the
        # count column, so a repeated floor value there must be marked or it reads as a measurement.
        # Judged against the count PRINTED (the marginal rung), which `saturation` (the floor rung's)
        # does not describe. The floor needs no sigma to recover: `n_floor * saturation` is
        # `[1/(I_p sigma^2)] * [I_p/ceiling] = 1/(ceiling sigma^2)`, exactly.
        at_floor = "=" if count <= _FLOOR_TOLERANCE * row.n_floor * row.saturation else ""
        cell = "--" if source == "unmeasured" else f"{count:.3g}{mark}{at_floor}"
        efficiency = "--" if row.pose_efficiency is None else f"{row.pose_efficiency:.3f}"
        lines.append(
            f"| {row.name} | {row.mass_kda:.0f} | {row.motion} | {row.ca_rmsd:.2f} | "
            f"{row.d2_known_pose:.3g} | {efficiency} | {row.saturation:.2f}{saturated} | "
            f"{row.n_floor:.3g} | {cell} | "
            f"{verdict(row, budget=budget)} |"
        )
    return "\n".join(lines)


def assembly_option(value: str) -> str | None:
    """An `--assembly` command-line value, as `align_conformers` wants it.

    The default must stay `"1"`: at `None` the aligner superposes every copy at once and returns a
    believable C-alpha RMSD with every guard satisfied. Drivers call `add_assembly_argument` rather
    than spelling the option. See docs/rationale/structures.md.

    Args:
        value: The text as typed. `"none"` or `"null"`, in any case, asks for the deposited
            coordinates, which is right only when they are already the whole molecule.

    Returns:
        The assembly id as a string, or `None` for the deposit.
    """
    return None if value.strip().lower() in ("none", "null") else value


def add_assembly_argument(parser: Any) -> Any:
    """Give a driver's parser the `--assembly` option, spelled once for all of them.

    So `default="1"` lives in one file; see `assembly_option`.

    Args:
        parser: An `argparse` parser or argument group.

    Returns:
        The action, for a caller that wants to reach into it.
    """
    return parser.add_argument(
        "--assembly", default="1", type=assembly_option, help="biological assembly to build; 'none' for the deposit"
    )
