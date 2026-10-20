"""Two conformers of one molecule, put in a common frame.

Every rung reads `mu_A - mu_B`, so the two states must share a frame and an atom set. Neither holds
for two deposited structures, and both failures are silent: a rigid frame displacement reads as an
enormous discriminability, and atoms one crystal resolved and the other did not read as composition
the pose search can exploit. The pair is centered once, on the two states averaged. An allowlisted
ligand is exempt from the correspondence (a ligand in one state and not the other defines a ligation
state); everything else is matched by label and cut to the intersection. See docs/rationale/structures.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import gemmi
import mmdf
import numpy as np
import pandas as pd

from cryodisc.forward.cryojax_model import CryoJAXModel, pad_to_common_atom_count
from cryodisc.microscope import CollectionPlan

UNSET: Any = object()
"""Sentinel for a per-entry setting that was not given, so `None` stays a MEANINGFUL value.

`assembly=None` asks for the deposited coordinates and `chains=None` for every chain, so neither can
double as "not passed": `assembly_b=None` must be able to mean "state B is already the whole molecule".
"""

DEFAULT_ASSEMBLY = "1"
"""The assembly `align_conformers` builds when nothing asks for another one.

Named rather than left as a signature default, since the per-entry parameters push the real default
into `_per_entry`. Every driver's `--assembly` must agree with THIS, and the agreement is tested.
"""

ChainIdKind = Literal["auth_asym_id", "label_asym_id"]
"""Which of mmCIF's two chain namings a caller's chain ids are written in.

`mmdf` reads the author chain, so `"auth_asym_id"` indexes directly and is the default. BioEmu names
`label_asym_id`, which needs `label_subchain_atoms` to resolve and is NOT a chain name at all."""

_KEY = ["copy", "chain", "residue_id", "atom"]
_XYZ = ["x", "y", "z"]

_MIN_SHARED_FRACTION = 0.75
"""Below this fraction of the larger atom list, a label match is a correspondence failure.

Two independently solved conformers share nearly every atom. Around half is the signature of one
entry's assembly against the other's asymmetric unit, which still superposes to a believable RMSD.
"""

_MIN_MATCHED_ALPHA_CARBONS = 20
"""Fewer matched C-alpha than this and the superposition is not determined enough to trust."""

HYDROGEN = ("H", "D")
"""Elements dropped before two entries are matched, unless a caller asks for them.

Whether a deposition MODELS hydrogens is a property of the method, not the molecule, and it wrecks the
correspondence guard when two entries disagree. Deuterium goes with it. See docs/rationale/structures.md.
"""


@dataclass(frozen=True)
class Frame:
    """The rigid map from one deposited structure's own coordinates into the pair's shared frame.

    A deposited MAP is in its model's frame, not the pair's; each model was fitted into its own map, so
    the model-to-model superposition IS the map-to-map transform.

    Args:
        rotation: (3, 3), applied about `source_center`. Never a reflection; see `_kabsch`.
        source_center: (3,) point in the source frame the rotation is taken about, in A.
        target_center: (3,) where that point lands in the shared frame, in A.

    Rationale: docs/rationale/structures.md
    """

    rotation: np.ndarray
    source_center: np.ndarray
    target_center: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Move coordinates from the source frame into the shared one.

        Args:
            points: (n, 3) positions in A, ordered (x, y, z) as a structure file records them.

        Returns:
            (n, 3) positions in the shared, origin-centered frame the templates are rendered in.
        """
        return (np.asarray(points, float) - self.source_center) @ self.rotation.T + self.target_center


@dataclass(frozen=True)
class Alignment:
    """What the superposition did, and what it threw away to do it.

    Args:
        n_matched: Matched C-alpha the superposition used.
        rmsd: C-alpha RMSD after superposition, in A. The size of the conformational change; tens of A
            is a correspondence failure, not a large motion.
        n_atoms: CORRESPONDING atoms per state, equal by construction.
        n_dropped: Atoms dropped from each state, `(A, B)`, those the other did not resolve or resolved
            as a different element. An allowlisted ligand the other lacks is NOT dropped.
        n_state_specific: Allowlisted ligand atoms kept for each state alone, `(A, B)`. Zero unless
            `ligands` named a residue differing between the states; read it before believing a ligation
            state reached the templates.
        radius: Distance from the shared center to the furthest atom kept, in A, state-specific ligands
            included.
        frame_a: Transform taking state A's deposited coordinates into the shared frame.
        frame_b: The same for state B, which carries the rotation.
        shared_fraction: Share of the larger atom list the two entries had in common, recorded whether
            or not it cleared the threshold.
    """

    n_matched: int
    rmsd: float
    n_atoms: int
    n_dropped: tuple[int, int]
    n_state_specific: tuple[int, int]
    radius: float
    frame_a: Frame
    frame_b: Frame
    shared_fraction: float = 1.0

    def minimum_box_size(self, pixel_size: float) -> int:
        """Smallest even box that holds the molecule at this pixel size.

        A floor, not a recommendation: a box at exactly this size aliases the CTF's delocalization
        back onto the particle.

        Args:
            pixel_size: Pixel size in A.

        Returns:
            The box size in pixels.
        """
        return 2 * int(np.ceil(self.radius / pixel_size))

    def projection_box_size(self, pixel_size: float, box_size: int, margin: float = 8.0) -> int:
        """Plane the atom renderer evaluates on, before zero-padding to the image box.

        An image box holds the CTF's delocalization as well as the molecule, so most of what the erf
        rasterization evaluates is empty. Density outside this plane is lost, which is what the margin
        is for. Parity follows the image box. See docs/rationale/structures.md.

        Args:
            pixel_size: Pixel size in A.
            box_size: Image size in pixels the projection is padded to.
            margin: Room beyond the furthest atom, in A, holding the Gaussian tails.

        Returns:
            The plane in pixels, never wider than `box_size`.
        """
        span = 2 * int(np.ceil((self.radius + margin) / pixel_size))
        return min(span + (int(box_size) - span) % 2, int(box_size))


@dataclass(frozen=True)
class ConformerPair:
    """Two conformers sharing a frame, a center, and their corresponding atoms.

    Args:
        state_a: Atoms of state A, in the `mmdf` schema, coordinates in the shared frame. The first
            `alignment.n_atoms` rows are the corresponding set; anything after is an allowlisted ligand
            this state has and the other does not.
        state_b: Atoms of state B, whose first `alignment.n_atoms` rows are the same atoms in the same
            order as state A's. The two tables are NOT the same length when a ligation state differs.
        alignment: What the superposition did.
    """

    state_a: pd.DataFrame
    state_b: pd.DataFrame
    alignment: Alignment

    def models(
        self, plan: CollectionPlan, box_size: int = 128, *, form_factor: str = "peng"
    ) -> tuple[CryoJAXModel, CryoJAXModel]:
        """Build both states' forward models on the grid a planned collection implies.

        Pixel size and voltage come from the plan, the two disagreeing being invisible. A caller with
        no collection wants `models_on_grid` instead.

        Args:
            plan: The collection these templates are for.
            box_size: Image size in pixels. `alignment.minimum_box_size` is the floor.

        Returns:
            The models for state A and state B, in that order.
        """
        return self.models_on_grid(
            box_size=box_size,
            pixel_size=plan.pixel_size,
            voltage_in_kv=plan.voltage_in_kv,
            form_factor=form_factor,
        )

    def models_on_grid(
        self, *, box_size: int, pixel_size: float, voltage_in_kv: float = 300.0, form_factor: str = "peng"
    ) -> tuple[CryoJAXModel, CryoJAXModel]:
        """Build both states' forward models on a stated grid, with no collection behind it.

        For rendering the two states to be LOOKED at rather than imaged -- a picking template, a
        reference map, a projected diameter.

        Args:
            box_size: Image size in pixels. `alignment.minimum_box_size` is the floor.
            pixel_size: A/px of that box.
            voltage_in_kv: Accelerating voltage the wavelength is taken from; must not disagree with the
                plan a `resolution` cutoff came from.
            form_factor: `"peng"`, or `"z"` for E7's form-factor ablation.

        Returns:
            The models for state A and state B, padded to a common atom count for `particles.sim`. Both
            render on ONE projection plane, sized from `alignment.radius` -- a plane per state would
            truncate them differently, landing in `d^2` as signal.

        Rationale: docs/rationale/structures.md
        """
        build = partial(
            CryoJAXModel.from_structure,
            box_size=int(box_size),
            pixel_size=float(pixel_size),
            voltage_in_kilovolts=float(voltage_in_kv),
            center=False,
            projection_shape=self.alignment.projection_box_size(float(pixel_size), int(box_size)),
            form_factor=form_factor,
        )
        model_a, model_b = pad_to_common_atom_count((build(self.state_a), build(self.state_b)))
        return model_a, model_b


def _per_entry(shared: Any, only_a: Any, only_b: Any, name: str, default: Any) -> tuple[Any, Any]:
    """One setting per entry, from a shared value or two separate ones, refusing both at once.

    Separate parameters rather than a two-sequence, which cannot be disambiguated; the DEFAULT is
    applied here rather than in the signature. See docs/rationale/structures.md.

    Args:
        shared: The value meant for both entries, or `UNSET`.
        only_a: State A's own value, overriding `shared`.
        only_b: State B's own value.
        name: Parameter name, for the message.
        default: What both entries get when nothing was passed at all.

    Returns:
        `(for_a, for_b)`.

    Raises:
        ValueError: If a shared value and a per-entry one are both given.
    """
    if shared is not UNSET and (only_a is not UNSET or only_b is not UNSET):
        raise ValueError(
            f"pass either {name} for both entries or {name}_a/{name}_b for one each, not both: "
            f"got {name}={shared!r} beside {name}_a={only_a!r} and {name}_b={only_b!r}"
        )
    fallback = default if shared is UNSET else shared
    return (
        fallback if only_a is UNSET else only_a,
        fallback if only_b is UNSET else only_b,
    )


def align_conformers(
    path_a: str | Path,
    path_b: str | Path,
    *,
    assembly: str | int | None = UNSET,
    assembly_a: str | int | None = UNSET,
    assembly_b: str | int | None = UNSET,
    chains: tuple[str, ...] | None = UNSET,
    chains_a: tuple[str, ...] | None = UNSET,
    chains_b: tuple[str, ...] | None = UNSET,
    chain_id_kind: ChainIdKind = "auth_asym_id",
    labels_a: tuple[str, ...] | None = None,
    labels_b: tuple[str, ...] | None = None,
    ligands: tuple[str, ...] = (),
    model: int = 1,
    hydrogens: bool = False,
    superpose: bool = True,
    offset: tuple[float, float, float] | None = None,
    min_shared_fraction: float = _MIN_SHARED_FRACTION,
) -> ConformerPair:
    """Read two structures, restrict them to their shared atoms, and put them in one frame.

    Args:
        path_a: Structure of state A, in any format `gemmi` reads.
        path_b: Structure of state B.
        assembly: Biological assembly to build, named as the file names it, usually `"1"`. `None` takes
            the deposited coordinates, right only when they already are the whole molecule. An `int` is
            stringified.
        assembly_a: State A's own assembly, overriding `assembly`.
        assembly_b: State B's own assembly.
        chains: Author chains to keep, applied before the assembly is expanded. `None` keeps every chain.
        chains_a: State A's own chains, overriding `chains`.
        chains_b: State B's own chains.
        chain_id_kind: Which naming `chains` is written in, for BOTH entries. A label is resolved to its
            subchain's residues, not the author chain holding it (see `label_subchain_atoms`).
        labels_a: Chain names to rename state A's assembly-generated copies onto, in `(copy, chain)`
            order (see `_relabel_copies`) -- matches an entry that GENERATES a tetramer against one that
            deposits it.
        labels_b: The same for state B.
        ligands: Residue names of non-water HETATM to KEEP, e.g. `("HEM",)` or `("HEM", "CMO", "OXY")`.
            An allowlist; a named residue is exempt from the correspondence and kept by whichever state
            has it. Read `Alignment.n_state_specific` for how many entered that way.
        model: Model number, which matters only for an NMR ensemble.
        hydrogens: Keep modeled hydrogens. Off; see `HYDROGEN`.
        superpose: Superpose B onto A before centering. On for two INDEPENDENT depositions; off for two
            structures ALREADY in one frame, where the superposition is the error.
        offset: Where to put the pair's shared center, in A, instead of the origin. `None` centers it,
            NOT right for images generated from volumes carrying their own origin.

    Returns:
        The pair, in the shared frame and centered on the origin unless `offset` moved it. The center is
        taken over the CORRESPONDING atoms, so an allowlisted ligand in one state alone does not pull
        the frame toward that state. With `superpose=False` both states carry the frame they arrived in.

    Raises:
        ValueError: If the two atom lists overlap too little to be the same molecule expressed the same
            way, or if too few C-alpha match to superpose on.

    Rationale: docs/rationale/structures.md
    """
    for_a, for_b = _per_entry(assembly, assembly_a, assembly_b, "assembly", default=DEFAULT_ASSEMBLY)
    chains_for_a, chains_for_b = _per_entry(chains, chains_a, chains_b, "chains", default=None)
    named_a = None if for_a is None else str(for_a)
    named_b = None if for_b is None else str(for_b)
    allowlist = _ligand_names(ligands)
    atoms_a = _read_assembly(path_a, named_a, chains_for_a, allowlist, model, labels_a, hydrogens, chain_id_kind)
    atoms_b = _read_assembly(path_b, named_b, chains_for_b, allowlist, model, labels_b, hydrogens, chain_id_kind)
    shared_a, shared_b, extra_a, extra_b, shared_fraction = _shared_atoms(
        atoms_a, atoms_b, path_a, path_b, allowlist, min_shared_fraction
    )

    is_alpha_carbon = (shared_a["atom"] == "CA").to_numpy()
    n_matched = int(is_alpha_carbon.sum())
    if n_matched < _MIN_MATCHED_ALPHA_CARBONS:
        raise ValueError(
            f"only {n_matched} matched C-alpha between {path_a} and {path_b}, too few to superpose "
            "on. A structure with no protein backbone needs a superposition this does not implement."
        )

    coords_a = shared_a[_XYZ].to_numpy()
    coords_b = shared_b[_XYZ].to_numpy()
    if superpose:
        rotation, centroid_b, centroid_a = _kabsch(coords_b[is_alpha_carbon], coords_a[is_alpha_carbon])
        coords_b = (coords_b - centroid_b) @ rotation.T + centroid_a
    else:
        # the identity, so `frame_b` composes to the same shared centering `frame_a` gets and the
        # rmsd below is the motion AS DEPOSITED rather than what is left after rotating some away
        rotation, centroid_b, centroid_a = np.eye(3), np.zeros(3), np.zeros(3)
    rmsd = float(np.sqrt(((coords_b[is_alpha_carbon] - coords_a[is_alpha_carbon]) ** 2).sum(1).mean()))

    # The pair's own center, not each state's: cryoJAX would center the two separately and the
    # difference between their centers of mass is real conformational change.
    mass = shared_a["atomic_weight"].to_numpy()[:, None]
    center = 0.5 * (coords_a + coords_b).T @ mass / mass.sum()
    shift = np.zeros(3) if offset is None else np.asarray(offset, float).ravel()
    if shift.shape != (3,):
        raise ValueError(f"offset must be three numbers in A, got shape {shift.shape}")
    center = center.ravel() - shift
    coords_a, coords_b = coords_a - center, coords_b - center

    # the two arrival frames composed with the shared centering, so a caller holding a deposited MAP can
    # place it. State A moves too (the pair is centered on both averaged); the ligands are placed by these.
    frame_a = Frame(np.eye(3), center, np.zeros(3))
    frame_b = Frame(rotation, centroid_b, centroid_a - center)
    state_a = _placed(shared_a, coords_a, extra_a, frame_a)
    state_b = _placed(shared_b, coords_b, extra_b, frame_b)
    return ConformerPair(
        state_a=state_a,
        state_b=state_b,
        alignment=Alignment(
            n_matched=n_matched,
            rmsd=rmsd,
            n_atoms=len(shared_a),
            n_dropped=(len(atoms_a) - len(state_a), len(atoms_b) - len(state_b)),
            n_state_specific=(len(extra_a), len(extra_b)),
            # measured from the ORIGIN, which an offset pair is deliberately not centered on, so the
            # box floor this implies still contains the molecule wherever the caller put it
            radius=max(_radius(state_a), _radius(state_b)),
            frame_a=frame_a,
            frame_b=frame_b,
            shared_fraction=shared_fraction,
        ),
    )


def _placed(shared: pd.DataFrame, coords: np.ndarray, extra: pd.DataFrame, frame: Frame) -> pd.DataFrame:
    """One state's atoms in the shared frame: the corresponded ones, then its own ligands.

    The corresponding atoms carry coordinates already computed; the exempt ones are moved by the frame.

    Args:
        shared: This state's corresponding atoms, as `_shared_atoms` returned them.
        coords: (n, 3) where those atoms belong in the shared frame, in A, in the same row order.
        extra: This state's allowlisted ligand atoms with no counterpart, in deposited coordinates.
        frame: This state's transform into the shared frame.

    Returns:
        The atoms, corresponded rows first so `alignment.n_atoms` indexes the boundary.
    """
    placed = shared.copy()
    placed[_XYZ] = coords
    if extra.empty:
        return placed
    moved = extra.copy()
    moved[_XYZ] = frame.apply(extra[_XYZ].to_numpy())
    return pd.concat([placed, moved], ignore_index=True)


def _radius(atoms: pd.DataFrame) -> float:
    """Distance from the origin to this state's furthest atom, in A."""
    return float(np.linalg.norm(atoms[_XYZ].to_numpy(), axis=1).max())


def _ligand_names(ligands: tuple[str, ...]) -> tuple[str, ...]:
    """The allowlist as residue names are stored, so the two places that read it cannot disagree.

    `_read_assembly` admits these residues and `_shared_atoms` exempts them from the correspondence.
    Normalizing separately is one edit away from an allowlist that admits a ligand and then discards it.
    """
    return tuple(name.strip().upper() for name in ligands)


def label_subchain_atoms(path: str | Path, labels: Sequence[str], model: int = 1) -> tuple[tuple[str, int, str], ...]:
    """`(author chain, residue id, residue name)` of every residue in the named mmCIF `label_asym_id` subchains.

    A label is not interchangeable with the author chain holding it: mmCIF splits one author chain into
    polymer, ligand and water subchains, so resolving a label to its chain NAME would return the whole
    subunit. Only an mmCIF carries `label_asym_id`; a missing label refuses rather than falling through.

    Args:
        path: The structure, which must be an mmCIF.
        labels: The `label_asym_id` values to resolve.
        model: Model number, matching `_read_assembly`'s.

    Returns:
        The `(chain, residue_id, residue name)` keys `mmdf` indexes those atoms by, in file order and
        deduplicated. The residue NAME is part of the key: two subchains of one author chain may share
        a residue number, and `(chain, residue_id)` alone would silently widen the selection.

    Raises:
        ValueError: If the file is not an mmCIF, if it has no such model, or if any label is absent.
    """
    if Path(path).suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError(
            f"{path} carries no label_asym_id to resolve {tuple(labels)} against, being a PDB file. "
            "Fetch the .cif, or name the author chains and leave chain_id_kind at 'auth_asym_id'."
        )
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if not 0 < model <= len(structure):
        raise ValueError(f"{path} has {len(structure)} model(s), so model {model} is not one of them")
    chains = list(structure[model - 1])
    wanted: set[str] = set(labels)
    present = {sub.subchain_id() for chain in chains for sub in chain.subchains()}
    if missing := sorted(wanted - present):
        raise ValueError(f"{path} has no label_asym_id {missing} in model {model}; it has {sorted(present)}")
    keys: list[tuple[str, int, str]] = []
    for chain in chains:
        for subchain in chain.subchains():
            if subchain.subchain_id() not in wanted:
                continue
            # `seqid.num` is optional in gemmi's model; a residue without one cannot be keyed against
            # `mmdf`'s `residue_id` and would silently widen the selection, so it is skipped
            keys.extend(
                (chain.name, int(residue.seqid.num), residue.name)
                for residue in subchain
                if residue.seqid.num is not None
            )
    return tuple(dict.fromkeys(keys))


def _read_assembly(
    path: str | Path,
    assembly: str | None,
    chains: tuple[str, ...] | None,
    ligands: tuple[str, ...],
    model: int,
    labels: tuple[str, ...] | None = None,
    hydrogens: bool = False,
    chain_id_kind: ChainIdKind = "auth_asym_id",
) -> pd.DataFrame:
    """Atoms of one structure's biological assembly, keyed so two entries can be matched.

    `ligands` must already be normalized by `_ligand_names`. `chain_id_kind` says which naming
    `chains` is written in; see `label_subchain_atoms` for why the two cannot be tried in turn.
    """
    atoms = mmdf.read(str(path))
    atoms = atoms[atoms["model"] == model]
    atoms = atoms[(atoms["heteroatom_flag"] == "A") | atoms["residue"].isin(list(ligands))]
    if not hydrogens:
        atoms = atoms[~atoms["element"].str.strip().str.upper().isin(HYDROGEN)]
    if chains is not None:
        if chain_id_kind == "label_asym_id":
            keys = set(label_subchain_atoms(path, chains, model))
            index = pd.MultiIndex.from_arrays([atoms["chain"], atoms["residue_id"], atoms["residue"]])
            atoms = atoms[index.isin(keys)]
        else:
            atoms = atoms[atoms["chain"].isin(list(chains))]

    copies = []
    for copy, (generated_chains, rotation, translation) in enumerate(_assembly_operators(path, assembly)):
        subset = atoms if generated_chains is None else atoms[atoms["chain"].isin(generated_chains)]
        subset = subset.copy()
        subset[_XYZ] = subset[_XYZ].to_numpy() @ rotation.T + translation
        subset["copy"] = copy
        copies.append(subset)

    expanded = pd.concat(copies, ignore_index=True)
    if expanded.empty:
        raise ValueError(
            f"{path}: no atoms left after selecting model {model}, chains {chains} and assembly {assembly!r}"
        )
    if labels is not None:
        expanded = _relabel_copies(expanded, labels, path)
    # First altloc wins, which is also what makes the key unique and so what lets the two states be
    # matched by it. cryoJAX would otherwise place a partial-occupancy atom at full weight twice.
    return expanded.drop_duplicates(subset=_KEY, keep="first")


def _relabel_copies(expanded: pd.DataFrame, labels: tuple[str, ...], path: str | Path) -> pd.DataFrame:
    """Assembly-generated copies renamed onto the chains a second entry deposits.

    A generator does NOT rename what it produces, so its `(copy, chain)` pairs are taken in ascending
    order and renamed to `labels` in that order, with `copy` collapsed to zero. Relabeling within the
    operator's orbit cannot change the answer (the superposition absorbs it), so a wrong label is caught
    by the RMSD guard.

    Args:
        expanded: Atoms of one entry after its assembly has been built.
        labels: Chain name for each `(copy, chain)` pair, in ascending order of that pair.
        path: The file, for the message.

    Returns:
        The same atoms with `chain` renamed and `copy` zeroed.

    Raises:
        ValueError: If the number of labels does not match the number of pairs.
    """
    pairs = sorted({(int(copy), str(chain)) for copy, chain in zip(expanded["copy"], expanded["chain"], strict=True)})
    if len(pairs) != len(labels):
        raise ValueError(
            f"{path}: the assembly has {len(pairs)} (copy, chain) pairs {pairs} and {len(labels)} labels "
            f"{list(labels)} were given. One label per pair, in that order."
        )
    renamed = dict(zip(pairs, labels, strict=True))
    relabeled = expanded.copy()
    keys = list(zip(relabeled["copy"].astype(int), relabeled["chain"].astype(str), strict=True))
    relabeled["chain"] = [renamed[key] for key in keys]
    relabeled["copy"] = 0
    return relabeled


def _assembly_operators(
    path: str | Path, assembly: str | None
) -> list[tuple[list[str] | None, np.ndarray, np.ndarray]]:
    """The `(chains, rotation, translation)` of each copy the assembly is built from.

    A generator names its parts as author chains in a PDB file but as label subchains in an mmCIF
    one, and gemmi reports whichever the file used, leaving the other empty. `mmdf` only carries the
    author chain, so subchains are mapped back onto it -- several subchains to one chain, since mmCIF
    splits a chain's polymer, ligands and waters apart.
    """
    if assembly is None:
        return [(None, np.eye(3), np.zeros(3))]
    structure = gemmi.read_structure(str(path))
    author_chain = {sub.subchain_id(): chain.name for chain in structure[0] for sub in chain.subchains()}
    for candidate in structure.assemblies:
        if candidate.name != assembly:
            continue
        operators: list[tuple[list[str] | None, np.ndarray, np.ndarray]] = [
            (
                sorted({author_chain.get(name, name) for name in (generator.chains or generator.subchains)}),
                np.array(operator.transform.mat.tolist()),
                np.array(operator.transform.vec.tolist()),
            )
            for generator in candidate.generators
            for operator in generator.operators
        ]
        if not operators:
            raise ValueError(f"{path} defines assembly {assembly!r} but it generates no copies")
        return operators
    raise ValueError(
        f"{path} defines no assembly {assembly!r}; it has {[a.name for a in structure.assemblies]}. "
        "Pass assembly=None if the deposited coordinates are the whole molecule."
    )


def _shared_atoms(
    atoms_a: pd.DataFrame,
    atoms_b: pd.DataFrame,
    path_a: str | Path,
    path_b: str | Path,
    ligands: tuple[str, ...] = (),
    min_shared_fraction: float = _MIN_SHARED_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Both structures cut to the atoms they agree on, plus each one's own allowlisted ligands.

    Args:
        atoms_a: State A's assembly, from `_read_assembly`.
        atoms_b: State B's.
        path_a: State A's file, for the message.
        path_b: State B's.
        ligands: Residue names exempt from the correspondence, normalized by `_ligand_names`.
        min_shared_fraction: Overlap below which the pair is refused. Lower it to score a partial
            correspondence; zero overlap is refused anyway.

    Returns:
        `(shared_a, shared_b, extra_a, extra_b, fraction)`. The two `shared` tables are the same atoms
        in the same order; each `extra` is that state's allowlisted ligand atoms which did not
        correspond. `fraction` is the overlap, recorded whether or not it cleared the threshold.

    Raises:
        ValueError: If the two atom lists share nothing, or overlap less than `min_shared_fraction`.
    """
    indexed_a, indexed_b = atoms_a.set_index(_KEY), atoms_b.set_index(_KEY)
    shared = indexed_a.index.intersection(indexed_b.index)
    fraction = len(shared) / max(len(indexed_a), len(indexed_b), 1)
    # A partial correspondence is refused at `min_shared_fraction` (a judgement that can be relaxed);
    # no correspondence at all is refused whatever it is set to, leaving no difference map to score.
    if not len(shared) or fraction < min_shared_fraction:
        # Two failures want opposite responses, so the message says which: counts agreeing while nothing
        # corresponds is a numbering offset; counts differing as much as the overlap is a different construct.
        counts_agree = min(len(indexed_a), len(indexed_b)) / max(len(indexed_a), len(indexed_b), 1) > 0.9
        diagnosis = (
            "the two atom counts agree to within 10% while almost nothing corresponds, which is a "
            "residue-NUMBERING mismatch rather than a different construct: the same molecule is "
            "keyed differently in the two entries, and neither `chains` nor `assembly` can fix it"
            if counts_agree and fraction < 0.05
            else "most often one entry's assembly against part of the other's -- name `chains` to "
            "cut both to a common subunit, or check `assembly`"
        )
        raise ValueError(
            f"{path_a} and {path_b} share only {len(shared)} atoms of {len(indexed_a)} and "
            f"{len(indexed_b)} ({fraction:.0%}), so they do not express the same molecule the same "
            f"way -- {diagnosis}. Superposing them anyway returns a believable RMSD for the wrong "
            "correspondence."
        )
    # Indexing both by one `shared` is what puts the rows in correspondence; a mutated position keeps
    # its atom names but changes element, and those atoms are dropped rather than compared.
    selected_a, selected_b = indexed_a.loc[shared], indexed_b.loc[shared]
    same_element = (selected_a["atomic_number"] == selected_b["atomic_number"]).to_numpy()
    # An allowlisted residue must also agree by NAME: the key carries the residue NUMBER, so two
    # different ligands at the same numbered site would otherwise match. Each state keeps its own below.
    allowlisted = _is_allowlisted(selected_a, ligands) | _is_allowlisted(selected_b, ligands)
    same_residue = selected_a["residue"].to_numpy() == selected_b["residue"].to_numpy()
    corresponds = same_element & (same_residue | ~allowlisted)
    shared_a, shared_b = selected_a[corresponds], selected_b[corresponds]
    return (
        shared_a.reset_index(),
        shared_b.reset_index(),
        _state_specific(indexed_a, shared_a.index, ligands),
        _state_specific(indexed_b, shared_b.index, ligands),
        fraction,
    )


def _is_allowlisted(atoms: pd.DataFrame, ligands: tuple[str, ...]) -> np.ndarray:
    """Which rows are atoms of a residue the caller named in `ligands`."""
    return atoms["residue"].isin(list(ligands)).to_numpy()


def _state_specific(indexed: pd.DataFrame, matched: pd.Index, ligands: tuple[str, ...]) -> pd.DataFrame:
    """One state's allowlisted ligand atoms that the correspondence did not claim.

    Args:
        indexed: That state's whole assembly, keyed by `_KEY`.
        matched: Keys of the atoms that did correspond.
        ligands: Residue names exempt from the correspondence.

    Returns:
        The exempt atoms, in deposited coordinates, with `_KEY` back in the columns.
    """
    exempt = indexed[_is_allowlisted(indexed, ligands)]
    return exempt[~exempt.index.isin(matched)].reset_index()


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation taking `mobile` onto `reference`, with the two centroids it is applied about.

    Args:
        mobile: Points to move, (n, 3).
        reference: Points to move onto, (n, 3), in correspondence with `mobile`.

    Returns:
        `(rotation, mobile_centroid, reference_centroid)`, applied as
        `(mobile - mobile_centroid) @ rotation.T + reference_centroid`. Never a mirror -- the
        reflection SVD would otherwise admit is removed.
    """
    mobile_centroid, reference_centroid = mobile.mean(0), reference.mean(0)
    correlation = (mobile - mobile_centroid).T @ (reference - reference_centroid)
    left, _, right = np.linalg.svd(correlation)
    parity = np.sign(np.linalg.det(right.T @ left.T))
    return right.T @ np.diag([1.0, 1.0, parity]) @ left.T, mobile_centroid, reference_centroid
