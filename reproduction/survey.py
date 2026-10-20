"""The population the survey draws from, and the joins that turn four catalogs into one table.

Four catalogs supply the membership rule, each parser returning `(pairs, rejections)`: Kincore
(DFG / dihedral labels), GPCRdb (Active/Inactive `state`), DynDom (rotation angle, the motion-magnitude
axis `d^2` is plotted against) and BioEmu-benchmarks `multiconf` (a curated cross-check). Deduplicated
by UniProt accession; Kincore's AlphaFold models are dropped; a UniProt mass is the canonical chain,
not the particle. Every fetch arrives through an injected `Fetcher`.

Rationale: docs/rationale/preprocess.md
"""

import csv
import hashlib
import html
import io
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from typing import Any

from reproduction.record import read_payload

Fetcher = Callable[[str], str]
"""Fetches one URL and returns its body as text. The only network this module knows about."""

KINCORE_URL = "https://dunbrack.fccc.edu/kincore/static/downloads/fasta_with_labels/PK_labels_PDB.fasta"
"""Kincore's labeled kinase chains, one FASTA record per PDB chain."""

GPCRDB_URL = "https://gpcrdb.org/services/structure/"
"""GPCRdb's whole structure table as JSON.

`Other` is a fourth `state` value no summary of the source mentions; a parser assuming the three
documented states would silently drop it, so it is handled here rather than assumed away.
"""

DYNDOM_NR_URL = "https://dyndom.cmp.uea.ac.uk/dyndom/browse.do?type=NR"
"""DynDom's non-redundant listing as one HTML page. There is no bulk download.

Reached from `dyndomDatabases.jsp#NRD` -> "Browse Movements". The site's own links carry a
`;jsessionid=...` path parameter; the bare path works without one and does not stop working, so this
one omits it.
"""

DYNDOM_CLASSES_URL = "https://dyndom.cmp.uea.ac.uk/dyndom/interface/index.html"
"""DynDom's hinge/shear classification, a static page grouping NR runs under headings.

One fetch for the motion class of the whole population, rather than reading the class off each run page.
"""

DYNDOM_RUN_URL = "https://dyndom.cmp.uea.ac.uk/dyndom/dyndomRun.do?ddid={ddid}&action=view"
"""One DynDom run, carrying the rotation angle, translation and closure per domain pair.

Fetched per run, for the rows that survived the filter rather than the whole listing -- the listing
page has no angle on it, and the angle is what DynDom is here for.
"""

DYNDOM_TRANSLATION_URL = "https://dyndom.cmp.uea.ac.uk/dyndom/interface/translation{ddid}.html"
"""A static summary of one classified run, carrying its angle of rotation and nothing else.

The fallback for `DYNDOM_RUN_URL`, which answers 500 for some runs (the same ones on every attempt, a
server fault not a rate limit) while this static page answers 200. Its angle is the same quantity, so
it recovers the axis for those rows at the cost of the translation and the closure.
"""

SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
"""PDBe SIFTS residue-level mapping for one PDB entry, keyed by chain inside the response.

Per entry and not batched: the documented POST form of this endpoint answers 405, so a batch of
accessions is a batch of requests. A missing or obsolete entry answers 404, which is a rejection
with a reason and not an error -- DynDom's older runs cite entries that have since been superseded.
"""

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query={query}&fields=accession,id,protein_name,mass,organism_name&format=tsv&size=500"
)
"""UniProt search, TSV, accepting an OR'd batch of terms in one query.

Batched because the alternative is one request per accession for thousands of accessions. `size=500`
is the service's maximum page, so a batch is kept below it rather than paged.
"""

BIOEMU_URL = (
    "https://raw.githubusercontent.com/microsoft/bioemu-benchmarks/main/"
    "bioemu_benchmarks/assets/multiconf_benchmark_0.1/{category}/references.csv"
)
"""One BioEmu `multiconf` category's reference pairs. CDLA-2.0."""

BIOEMU_CATEGORIES = ("domainmotion", "crypticpocket", "oodval")
"""The `multiconf` categories that name TWO conformers.

`localunfolding` is excluded: its `references.csv` has a single `pdbidchain` column, so its cases are
one structure each and there is no pair to compare.
"""

ALPHAFOLD_PREFIX = "AF-"
"""How Kincore marks a predicted model. See the module docstring for why they are dropped."""

KINCORE_MISSING = "None"
"""What Kincore writes in a label column it has no value for -- the string, not an empty field."""

DYNDOM_MOTION = {
    "No Contact": "no_contact",
    "Interface-Preserving Movement": "shear",
    "Interface-Creating Movement": "hinge",
    "Mixed": "mixed",
}
"""DynDom's own heading text for each classification group, mapped to the name used here.

Interface-preserving and interface-creating correspond to the shear and hinge movements of Gerstein,
Lesk & Chothia (Biochemistry 33:6738, 1994), the vocabulary the panel already uses, so the
translation happens here once rather than in a plot label.
"""

MOTION_UNCLASSIFIED = "unclassified"
"""Motion class of an NR run the hinge/shear page does not list."""


READABLE_MOTION = {
    "no_contact": "no shared interface",
    "hinge": "hinge",
    "shear": "shear",
    "dfg_flip": "DFG flip",
    "dihedral_cluster": "loop / side-chain",
    "activation": "activation loop",
    "crypticpocket": "cryptic pocket",
    "domainmotion": "domain motion",
    "mixed": "mixed",
    MOTION_UNCLASSIFIED: "unclassified",
}
"""Motion classes in words a structural biologist reads without a key.

The single source of truth, so a class added to `DYNDOM_MOTION` or `BIOEMU_CATEGORIES` needs one
edit to become readable everywhere, and a missing class is missing from every panel at once rather
than from whichever figure was not updated.
"""

MIN_CLASS = 20
"""Smallest class a panel names, below which the per-class error bar is not worth drawing."""


@dataclass(frozen=True)
class Conformer:
    """One PDB chain in one labeled state.

    Args:
        pdb: PDB entry id, lower case.
        chain: Author chain id, or the empty string when the source named an entry and no chain.
            Empty rather than `None` so it is a legal dictionary key: some DynDom rows name a chain
            for one conformer only, and dropping them would discard pairs for a reason unrelated to
            the motion.
        state: The source's own label for the state, verbatim.
    """

    pdb: str
    chain: str
    state: str


@dataclass(frozen=True)
class Pair:
    """One candidate two-state pair, as a source described it.

    Args:
        source: Which catalog it came from.
        name: The source's own name for the protein, kept for a human reading the table.
        motion: Motion class, in this module's vocabulary.
        a: First conformer.
        b: Second conformer.
        hint: A UniProt entry name or accession the source itself carries, upper case, or `None`
            when it carries neither and the accession has to come from SIFTS.
        rotation_deg: Domain rotation in degrees, or `None` when the source has no such quantity.
            Filled for DynDom by `attach_dyndom_details` after the mass filter has run.
        extra: Source-specific fields worth carrying into the output -- resolution, cryo-EM flag,
            chain counts, the DynDom run id. Not interpreted here.
    """

    source: str
    name: str
    motion: str
    a: Conformer
    b: Conformer
    hint: str | None = None
    rotation_deg: float | None = None
    extra: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The pair as JSON, flattened so a reader indexes columns rather than nested objects."""
        return {
            "source": self.source,
            "name": self.name,
            "motion": self.motion,
            "state_a": self.a.state,
            "pdb_a": self.a.pdb,
            "chain_a": self.a.chain,
            "state_b": self.b.state,
            "pdb_b": self.b.pdb,
            "chain_b": self.b.chain,
            "rotation_deg": self.rotation_deg,
            **dict(self.extra or {}),
        }


@dataclass(frozen=True)
class Protein:
    """What UniProt says about one accession.

    Args:
        accession: The accession.
        entry_name: UniProt entry name, e.g. `EGFR_HUMAN`.
        protein_name: Recommended name, with the synonyms UniProt appends.
        organism: Organism, as UniProt writes it.
        mass_da: Mass of the canonical sequence in daltons. See the module docstring: this is one
            chain, not the imaged particle.
    """

    accession: str
    entry_name: str
    protein_name: str
    organism: str
    mass_da: int

    @property
    def mass_kda(self) -> float:
        """Mass in kilodaltons, which is the unit the mass window is expressed in."""
        return self.mass_da / 1000.0


@dataclass(frozen=True)
class SurveyRow:
    """One protein, with every two-state pair any source offered for it.

    Args:
        protein: The UniProt record the row is keyed by.
        pairs: The pairs, sorted by source then name so the row is stable between runs.
    """

    protein: Protein
    pairs: tuple[Pair, ...]

    @property
    def sources(self) -> tuple[str, ...]:
        """Which catalogs reached this protein, sorted. Length above one is a source overlap."""
        return tuple(sorted({pair.source for pair in self.pairs}))

    @property
    def motions(self) -> tuple[str, ...]:
        """Motion classes claimed for this protein, sorted."""
        return tuple(sorted({pair.motion for pair in self.pairs}))

    @property
    def rotation_deg(self) -> float | None:
        """The largest rotation angle any pair reports, or `None` when no pair has one.

        The largest and not the mean: the angle is the motion-magnitude axis, and a protein whose
        NR runs span 3 and 40 degrees has a 40-degree movement in it. The per-pair angles are in
        `pairs` for anyone who wants the spread.
        """
        angles = [pair.rotation_deg for pair in self.pairs if pair.rotation_deg is not None]
        return max(angles) if angles else None

    def as_dict(self) -> dict[str, Any]:
        """The row as JSON."""
        return {
            "accession": self.protein.accession,
            "entry_name": self.protein.entry_name,
            "protein_name": self.protein.protein_name,
            "organism": self.protein.organism,
            "mass_kda": round(self.protein.mass_kda, 3),
            "sources": list(self.sources),
            "motions": list(self.motions),
            "rotation_deg": self.rotation_deg,
            "n_pairs": len(self.pairs),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


@dataclass(frozen=True)
class DynDomRun:
    """The quantities read off one DynDom run.

    Args:
        ddid: DynDom's run id, which is the key the listing page hands out.
        rotation_deg: Largest domain-pair rotation on the page, in degrees.
        translation_a: Screw translation along the rotation axis for that domain pair, in angstroms.
            `None` when it came off the summary page, which does not report one.
        closure_percent: How much of the movement is closure rather than twist, in percent. `None` on
            the same terms as the translation.
        n_domain_pairs: How many domain pairs the run reports. Above one means the protein moves in
            more than one place and the single angle above is a summary of several. `None` from the
            summary page, which reports one angle without saying how many pairs produced it -- `None`
            rather than 1, because a row claiming one domain pair it did not read is worse than a row
            admitting it does not know.
        page: Which page the numbers came from, `run` or `summary`. Recorded because the two are not
            interchangeable: a summary-only row cannot have a translation, and a reader comparing
            translations across rows needs to know which rows those are.
    """

    ddid: str
    rotation_deg: float
    translation_a: float | None
    closure_percent: float | None
    n_domain_pairs: int | None
    page: str


def cached(fetcher: Fetcher, directory: str | Path) -> Fetcher:
    """Wrap a fetcher so each URL is fetched at most once, ever.

    The survey's joins are thousands of small requests against four public services, and a re-run
    that re-fetched them would be both slow and rude. Keyed by a hash of the full URL rather than by
    a name derived from it: two SIFTS URLs differ only in a four-character entry id, and two UniProt
    queries differ only in a batch of accessions hundreds of characters long, so any readable
    shortening of the URL collides -- and a colliding cache silently answers one query with
    another's body, which is the one failure mode that would not look like a failure.

    Args:
        fetcher: The fetcher to wrap.
        directory: Where bodies are written. Created if absent.

    Returns:
        A `Fetcher` reading from disk when it can and delegating when it cannot.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    def fetch(url: str) -> str:
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        # the host in the name so a human can see what a cache holds, the digest for identity
        host = re.sub(r"[^a-z0-9]+", "_", url.split("//", 1)[-1].split("/", 1)[0].lower())
        path = root / f"{host}-{digest}.txt"
        if path.exists():
            return path.read_text()
        body = fetcher(url)
        path.write_text(body)
        return body

    return fetch


def _strip_tags(fragment: str) -> str:
    """HTML fragment as single-spaced text, entities resolved.

    DynDom's pages are generated JSP with newlines and tabs inside every cell, and one cell holds
    `1s3i <br/> (A)`, so a parser that does not first flatten the whitespace is a parser matching
    against layout.

    Args:
        fragment: Any HTML.

    Returns:
        The text.
    """
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_kincore(text: str, *, min_cluster_chains: int = 1) -> tuple[list[Pair], list[str]]:
    """Kincore's labeled FASTA as one two-state pair per gene per axis.

    Two axes, because Kincore labels two things and both are a conformational state: the DFG spatial
    label and the dihedral cluster. A gene can yield both, and they are different motions of the same
    protein rather than duplicates. The representative chain of a label is the one with the LONGEST
    labeled sequence, ties breaking on the identifier. See docs/rationale/preprocess.md.

    Args:
        text: The FASTA body.
        min_cluster_chains: Chains a dihedral cluster needs before it counts as a state. 1 keeps
            every cluster Kincore labels; raising it demands a cluster be seen more than once, which
            is a way of not building a state out of a single deposition.

    Returns:
        `(pairs, rejections)`. Rejections carry one aggregate line for the chains dropped before
        grouping and one line per gene that yielded no pair -- a gene with many chains of one state
        only is the commonest case, and the reader should see that it is.
    """
    records = _fasta_records(text)
    predicted = [name for name in records if name.startswith(ALPHAFOLD_PREFIX)]
    rejections = []
    if predicted:
        rejections.append(
            f"kincore: dropped {len(predicted)} AlphaFold model chains (identifier starts "
            f"{ALPHAFOLD_PREFIX!r}); a predicted conformation cannot validate a measured claim"
        )

    genes: dict[str, dict[str, Any]] = {}
    malformed = 0
    for header, sequence in records.items():
        if header.startswith(ALPHAFOLD_PREFIX):
            continue
        fields = header.split("\t")
        if len(fields) < 6 or len(fields[0]) < 5:
            malformed += 1
            continue
        identifier, gene, entry_name, _group, spatial, dihedral = fields[:6]
        gene_state = genes.setdefault(gene, {"entry_name": entry_name, "spatial": {}, "dihedral": {}})
        conformer = (identifier[:4].lower(), identifier[4:], len(sequence), identifier)
        for axis, label in (("spatial", spatial), ("dihedral", dihedral)):
            if label != KINCORE_MISSING:
                gene_state[axis].setdefault(label, []).append(conformer)
    if malformed:
        rejections.append(f"kincore: dropped {malformed} records whose header did not carry six labeled fields")

    pairs = []
    for gene, state in sorted(genes.items()):
        hint = state["entry_name"].upper()
        emitted = []
        spatial = state["spatial"]
        if "DFGin" in spatial and "DFGout" in spatial:
            pairs.append(
                _kincore_pair(gene, hint, "dfg_flip", ("DFGin", spatial["DFGin"]), ("DFGout", spatial["DFGout"]))
            )
            emitted.append("dfg_flip")
        clusters = [(label, chains) for label, chains in state["dihedral"].items() if len(chains) >= min_cluster_chains]
        # by population and then by label, so the two states are the two best-supported ones and the
        # choice does not move when Kincore adds a chain to a cluster that was already smaller
        clusters.sort(key=lambda item: (-len(item[1]), item[0]))
        if len(clusters) >= 2:
            pairs.append(_kincore_pair(gene, hint, "dihedral_cluster", clusters[0], clusters[1]))
            emitted.append("dihedral_cluster")
        if not emitted:
            rejections.append(
                f"kincore {gene}: one state only -- spatial {sorted(spatial)}, "
                f"dihedral clusters {sorted(label for label, _ in clusters)}"
            )
    return pairs, rejections


def _fasta_records(text: str) -> dict[str, str]:
    """FASTA as `{header: sequence}`, headers verbatim and without their `>`.

    Args:
        text: The FASTA body.

    Returns:
        The records, in file order. A repeated header keeps the first sequence, which cannot happen
        in Kincore's file and would silently lengthen a representative if it did.
    """
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.setdefault(header, "".join(chunks))
            header, chunks = line[1:].rstrip(), []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.setdefault(header, "".join(chunks))
    return records


def _kincore_pair(
    gene: str,
    hint: str,
    motion: str,
    first: tuple[str, list[tuple[str, str, int, str]]],
    second: tuple[str, list[tuple[str, str, int, str]]],
) -> Pair:
    """One Kincore pair from two labeled chain sets, each reduced to its longest chain.

    Args:
        gene: Kincore's gene name.
        hint: UniProt entry name for the gene.
        motion: Motion class to record.
        first: `(label, chains)` for one state.
        second: `(label, chains)` for the other.

    Returns:
        The pair, carrying how many chains each state was chosen from -- a state supported by one
        chain and a state supported by 300 are not equally solid, and the count is the only thing in
        this file that says which is which.
    """
    (label_a, chains_a), (label_b, chains_b) = first, second
    best_a = max(chains_a, key=lambda item: (item[2], item[3]))
    best_b = max(chains_b, key=lambda item: (item[2], item[3]))
    return Pair(
        source="kincore",
        name=gene,
        motion=motion,
        a=Conformer(best_a[0], best_a[1], label_a),
        b=Conformer(best_b[0], best_b[1], label_b),
        hint=hint,
        extra={
            "n_chains_a": len(chains_a),
            "n_chains_b": len(chains_b),
            "domain_residues_a": best_a[2],
            "domain_residues_b": best_b[2],
        },
    )


_GPCRDB_ACTIVE = "Active"
"""GPCRdb's `state` value for the active conformation.

A constant and not a parameter: it is GPCRdb's own vocabulary, not a preference. The source also
carries `Intermediate` and `Other`, and the module docstring records why both are dropped -- a
partially-activated receptor on the active side of a two-state comparison is a different pair, not a
differently-spelled one.
"""

_GPCRDB_INACTIVE = "Inactive"
"""GPCRdb's `state` value for the inactive conformation. See `_GPCRDB_ACTIVE`."""


def parse_gpcrdb(records: Iterable[Mapping[str, Any]]) -> tuple[list[Pair], list[str]]:
    """GPCRdb's structure table as one Active/Inactive pair per receptor.

    The representative structure of a state is the best cryo-EM one, and an X-ray structure only if
    the state has no cryo-EM structure at all. Not an arbitrary preference: the paper is about what a
    cryo-EM particle can be told apart, and a state whose only representative is a crystal structure
    is a state whose conformation may be a lattice artifact. "Best" is lowest reported resolution,
    ties broken on the PDB code.

    `Intermediate` and `Other` states are ignored rather than mapped onto one of the two: GPCRdb
    means something specific by them, and folding an intermediate into Active would put a
    partially-activated receptor on the active side of a two-state comparison.

    Args:
        records: The decoded `services/structure/` list.

    Returns:
        `(pairs, rejections)`, one rejection per receptor that does not have both states.
    """
    receptors: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        protein = str(record.get("protein") or "")
        state = str(record.get("state") or "")
        if not protein or state not in (_GPCRDB_ACTIVE, _GPCRDB_INACTIVE):
            continue
        receptors.setdefault(protein, {}).setdefault(state, []).append(record)

    pairs, rejections = [], []
    for protein, states in sorted(receptors.items()):
        if _GPCRDB_ACTIVE not in states or _GPCRDB_INACTIVE not in states:
            rejections.append(f"gpcrdb {protein}: only {sorted(states)} of ({_GPCRDB_ACTIVE}, {_GPCRDB_INACTIVE})")
            continue
        chosen = {state: _best_gpcr_structure(entries) for state, entries in states.items()}
        pairs.append(
            Pair(
                source="gpcrdb",
                name=protein,
                motion="activation",
                a=Conformer(
                    str(chosen[_GPCRDB_INACTIVE]["pdb_code"]).lower(),
                    str(chosen[_GPCRDB_INACTIVE]["preferred_chain"]),
                    _GPCRDB_INACTIVE,
                ),
                b=Conformer(
                    str(chosen[_GPCRDB_ACTIVE]["pdb_code"]).lower(),
                    str(chosen[_GPCRDB_ACTIVE]["preferred_chain"]),
                    _GPCRDB_ACTIVE,
                ),
                hint=protein.upper(),
                extra={
                    "resolution_a": chosen[_GPCRDB_INACTIVE].get("resolution"),
                    "resolution_b": chosen[_GPCRDB_ACTIVE].get("resolution"),
                    "cryoem_a": _is_cryoem(chosen[_GPCRDB_INACTIVE]),
                    "cryoem_b": _is_cryoem(chosen[_GPCRDB_ACTIVE]),
                    "receptor_class": chosen[_GPCRDB_ACTIVE].get("class"),
                    # the mass caveat made concrete: an active-state receptor is usually imaged bound
                    # to a heterotrimeric G protein, so the imaged particle is ~85 kDa heavier
                    "signalling_protein_a": _signalling(chosen[_GPCRDB_INACTIVE]),
                    "signalling_protein_b": _signalling(chosen[_GPCRDB_ACTIVE]),
                    "n_structures_a": len(states[_GPCRDB_INACTIVE]),
                    "n_structures_b": len(states[_GPCRDB_ACTIVE]),
                },
            )
        )
    return pairs, rejections


def _is_cryoem(record: Mapping[str, Any]) -> bool:
    """Whether a GPCRdb structure was determined by cryo-EM.

    Args:
        record: One structure record.

    Returns:
        Whether its `type` is electron microscopy. Electron crystallography, which GPCRdb reports
        separately for two structures, is not counted -- it is a lattice method.
    """
    return str(record.get("type") or "") == "Electron microscopy"


def _signalling(record: Mapping[str, Any]) -> str | None:
    """The signaling partner GPCRdb reports for a structure, or `None`.

    Args:
        record: One structure record.

    Returns:
        The partner type, e.g. `G protein` or `Arrestin`.
    """
    partner = record.get("signalling_protein")
    return str(partner.get("type")) if isinstance(partner, Mapping) and partner.get("type") else None


def _best_gpcr_structure(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The representative structure of one receptor state.

    Args:
        entries: Structures GPCRdb lists for that state.

    Returns:
        Cryo-EM before X-ray, then lowest resolution, then lowest PDB code. A missing resolution
        sorts last rather than first, a null being an absent measurement and not a good one.
    """
    return min(
        entries,
        key=lambda record: (
            not _is_cryoem(record),
            record.get("resolution") if isinstance(record.get("resolution"), int | float) else float("inf"),
            str(record.get("pdb_code") or ""),
        ),
    )


def parse_dyndom_pairs(page: str) -> tuple[list[Pair], list[str]]:
    """DynDom's non-redundant listing as explicit conformer pairs.

    The listing is a table of `name | conformers | family | View`, where the conformers cell holds
    two PDB codes each optionally followed by a chain in parentheses. Some rows name a chain for one
    conformer only, or neither, so a parser that required both would discard rows for a reason
    unrelated to the motion. A missing chain becomes the empty string, which `sifts_chain_accessions`
    resolves to the entry's dominant accession.

    The rotation angle is NOT on this page. `parse_dyndom_run` reads it per run, keyed by the `ddid`
    recovered here.

    Args:
        page: The HTML body.

    Returns:
        `(pairs, rejections)`. A pair's motion is `MOTION_UNCLASSIFIED` until
        `apply_dyndom_classes` supplies one.
    """
    pairs, rejections = [], []
    for row in re.split(r"<tr\b", page)[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) < 4:
            continue
        name = _strip_tags(cells[0])
        conformers = re.findall(r"([0-9][A-Za-z0-9]{3})\s*(?:\(([A-Za-z0-9]+)\))?", _strip_tags(cells[1]))
        run = re.search(r"ddid=([A-Za-z0-9_]+)", row)
        if len(conformers) != 2 or run is None:
            rejections.append(f"dyndom {name or '?'}: listing row named {len(conformers)} conformers, expected 2")
            continue
        (pdb_a, chain_a), (pdb_b, chain_b) = conformers
        pairs.append(
            Pair(
                source="dyndom",
                name=name,
                motion=MOTION_UNCLASSIFIED,
                a=Conformer(pdb_a.lower(), chain_a or "", "conformer_1"),
                b=Conformer(pdb_b.lower(), chain_b or "", "conformer_2"),
                extra={"ddid": run.group(1), "family": _strip_tags(cells[2]) or None},
            )
        )
    return pairs, rejections


def parse_dyndom_classes(page: str) -> dict[str, str]:
    """The hinge/shear classification of each classified NR run.

    The page is one table per class, each under an `<h3>` heading, and each row links to
    `translation<DDID>.html`. So the class of a run is which heading it sits under, and the parse is
    a split on the headings rather than a per-row lookup.

    Args:
        page: The HTML body of `DYNDOM_CLASSES_URL`.

    Returns:
        `{ddid: motion}` using this module's vocabulary. Runs under a heading `DYNDOM_MOTION` does
        not know are omitted rather than guessed, so an added class shows up as unclassified rows
        rather than as a silently wrong label.
    """
    classes: dict[str, str] = {}
    # split on the whole opening tag so a section starts at the heading TEXT: leaving the tag's `>`
    # on the heading would defeat every comparison against `DYNDOM_MOTION`
    for section in re.split(r"<h3\b[^>]*>", page)[1:]:
        heading = _strip_tags(section.split("</h3>", 1)[0])
        motion = next((name for text, name in DYNDOM_MOTION.items() if heading.startswith(text)), None)
        if motion is None:
            continue
        for ddid in re.findall(r"translation([A-Za-z0-9_]+)\.html", section):
            classes.setdefault(ddid, motion)
    return classes


def apply_dyndom_classes(pairs: Sequence[Pair], classes: Mapping[str, str]) -> list[Pair]:
    """Label DynDom pairs with their hinge/shear class.

    Args:
        pairs: Pairs from `parse_dyndom_pairs`.
        classes: `{ddid: motion}` from `parse_dyndom_classes`.

    Returns:
        The pairs, with `motion` set where the classification covers them. Left
        `MOTION_UNCLASSIFIED` where it does not: a row whose class is unknown is not a row whose class
        is `mixed`.
    """
    labeled = []
    for pair in pairs:
        ddid = str((pair.extra or {}).get("ddid") or "")
        labeled.append(replace(pair, motion=classes.get(ddid, pair.motion)))
    return labeled


def parse_dyndom_run(ddid: str, page: str) -> DynDomRun | None:
    """The rotation angle and its companions off one DynDom run page.

    A run reports one row per moving domain pair, so a multi-domain protein has several angles. The
    largest is taken and the count is recorded: the axis this feeds is motion magnitude, and the
    largest movement is the magnitude of the movement, while the count is what warns a reader that
    the protein does not move in only one place.

    Args:
        ddid: The run id, recorded on the result.
        page: The HTML body.

    Returns:
        The run, or `None` when the page has no rotation angle on it -- an expired session or a run
        whose analysis failed both answer 200 with a page that simply does not have the number.
    """
    text = _strip_tags(page)
    angles = [float(value) for value in re.findall(r"Rotation Angle \(deg\)\s*([-\d.]+)", text)]
    if not angles:
        return None
    best = max(range(len(angles)), key=lambda index: angles[index])
    translations = re.findall(r"Translation \(A\)\s*([-\d.]+)", text)
    closures = re.findall(r"Closure \(%\)\s*([-\d.]+)", text)
    return DynDomRun(
        ddid=ddid,
        rotation_deg=angles[best],
        translation_a=float(translations[best]) if best < len(translations) else None,
        closure_percent=float(closures[best]) if best < len(closures) else None,
        n_domain_pairs=len(angles),
        page="run",
    )


def parse_dyndom_translation(ddid: str, page: str) -> DynDomRun | None:
    """The rotation angle off the static summary page, for a run whose full page will not serve.

    Some runs answer 500 on `DYNDOM_RUN_URL`, the same ones on every attempt, while their summary
    page answers 200. Without this fallback those rows reach the output with a null angle, which is to
    say they reach the motion-magnitude figure not at all.

    Args:
        ddid: The run id.
        page: The HTML body of `DYNDOM_TRANSLATION_URL`.

    Returns:
        The run with its angle and nothing else, or `None` when the page has no angle either.
    """
    found = re.search(r"Angle of Rotation\s*([-\d.]+)\s*degrees", _strip_tags(page))
    if found is None:
        return None
    return DynDomRun(
        ddid=ddid,
        rotation_deg=float(found.group(1)),
        translation_a=None,
        closure_percent=None,
        n_domain_pairs=None,
        page="summary",
    )


def attach_dyndom_details(rows: Sequence[SurveyRow], details: Mapping[str, DynDomRun]) -> list[SurveyRow]:
    """Fill in the rotation angle of every DynDom pair a caller fetched a run page for.

    Separate from parsing because the listing page has no angle and each angle is one more request:
    a caller resolves and filters first, then fetches run pages for the rows that survived, then
    calls this, rather than fetching the whole listing up front to keep a fraction of it.

    Args:
        rows: Rows to enrich.
        details: `{ddid: run}` for the runs that were fetched.

    Returns:
        The rows, with `rotation_deg`, translation, closure and domain-pair count on the DynDom
        pairs the details cover, and untouched elsewhere.
    """
    enriched = []
    for row in rows:
        pairs = []
        for pair in row.pairs:
            run = details.get(str((pair.extra or {}).get("ddid") or ""))
            if run is None:
                pairs.append(pair)
                continue
            pairs.append(
                replace(
                    pair,
                    rotation_deg=run.rotation_deg,
                    extra={
                        **dict(pair.extra or {}),
                        "translation_a": run.translation_a,
                        "closure_percent": run.closure_percent,
                        "n_domain_pairs": run.n_domain_pairs,
                        "rotation_page": run.page,
                    },
                )
            )
        enriched.append(replace(row, pairs=tuple(pairs)))
    return enriched


def parse_bioemu_references(text: str, *, category: str) -> tuple[list[Pair], list[str]]:
    """One BioEmu `multiconf` category's `references.csv` as pairs.

    The column ORDER differs between categories, so the parse is by header name. `test_case` is
    usually a UniProt accession, making this the one source that mostly needs no identifier join; a
    case id that is not an accession keeps its conformers and loses its hint, so it resolves through
    SIFTS like a DynDom row -- and it is kept out of the UniProt query, where one malformed term
    answers 400 for the whole batch and costs every other term its mass.

    Args:
        text: The CSV body.
        category: Which category, recorded as the motion class since that is what BioEmu's
            categories are.

    Returns:
        `(pairs, rejections)`. A row missing either conformer column is rejected by name.
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or ())
    if not {"pdbidchain_i", "pdbidchain_j", "test_case"} <= fields:
        return [], [f"bioemu {category}: references.csv has columns {sorted(fields)}, expected a pair and a test case"]

    pairs, rejections = [], []
    for row in reader:
        accession = (row.get("test_case") or "").strip().upper()
        first, second = _split_pdb_chain(row.get("pdbidchain_i")), _split_pdb_chain(row.get("pdbidchain_j"))
        if first is None or second is None or not accession:
            rejections.append(f"bioemu {category}: unusable row {dict(row)}")
            continue
        if not is_accession(accession):
            rejections.append(
                f"bioemu {category}: test case {accession!r} is not a UniProt accession, "
                f"resolving {first[0]}:{first[1]} through SIFTS instead"
            )
        pairs.append(
            Pair(
                source="bioemu",
                name=accession,
                motion=category,
                a=Conformer(first[0], first[1], "reference_i"),
                b=Conformer(second[0], second[1], "reference_j"),
                hint=accession if is_accession(accession) else None,
                # label_asym_id and not the author chain, per the benchmark's own README, so a
                # consumer reading the deposition has to map it rather than index it directly
                extra={"chain_id_kind": "label_asym_id", "ligand_name": row.get("ligand_name") or None},
            )
        )
    return pairs, rejections


def _split_pdb_chain(token: str | None) -> tuple[str, str] | None:
    """`7UBJ_A` as `("7ubj", "A")`.

    Args:
        token: The identifier, or `None`.

    Returns:
        `(pdb, chain)`, or `None` when the token is not a four-character code and a chain.
    """
    parts = (token or "").strip().split("_")
    if len(parts) != 2 or len(parts[0]) != 4 or not parts[1]:
        return None
    return parts[0].lower(), parts[1]


def sifts_chain_accessions(payload: Mapping[str, Any], pdb: str) -> dict[str, str]:
    """The UniProt accession of each author chain of one PDB entry, from a SIFTS mapping.

    A chain can map to more than one accession -- a chimera, a fusion construct, or a designed
    linker -- and the one that matters is the one covering most of the chain. So accessions are
    ranked by mapped residue count rather than refused as ambiguous, which would throw away real
    fusion constructs for being annotated honestly.

    The empty-string key holds the entry's dominant accession, for DynDom rows that name an entry
    without naming a chain.

    Args:
        payload: The decoded response for one entry.
        pdb: The entry id, used to find it inside the response.

    Returns:
        `{chain: accession}`, keyed by author chain id and also by `struct_asym_id`, plus `""`.
        Empty when the response holds no UniProt mapping at all -- a nucleic-acid-only entry, or one
        SIFTS has not processed.
    """
    entry = payload.get(pdb.lower()) or payload.get(pdb.upper()) or {}
    mappings = entry.get("UniProt", {}) if isinstance(entry, Mapping) else {}
    coverage: dict[str, dict[str, int]] = {}
    for accession, block in mappings.items():
        for mapping in block.get("mappings", ()):
            residues = max(int(mapping.get("unp_end", 0)) - int(mapping.get("unp_start", 0)) + 1, 0)
            # Deduplicate the keys: `chain_id` and `struct_asym_id` are often the same letter, and
            # counting this mapping's residues twice would let a lesser-covering accession outrank a
            # greater one. `dict` rather than `set` to keep the order, for a reader.
            chain = str(mapping.get("chain_id") or "")
            asym = str(mapping.get("struct_asym_id") or "")
            for key in dict.fromkeys(("", chain, asym)):
                coverage.setdefault(key, {})
                coverage[key][accession] = coverage[key].get(accession, 0) + residues
    # tie broken on the accession so the answer does not depend on dictionary order
    return {
        chain: max(sorted(options), key=lambda accession: options[accession])
        for chain, options in coverage.items()
        if options
    }


def is_accession(term: str) -> bool:
    """Whether a string is syntactically a UniProtKB accession.

    Checked before a term reaches a query rather than after, because UniProt validates the
    `accession` filter and answers 400 for the WHOLE query when one value is malformed -- so a single
    bad identifier costs every other term in its batch its mass, showing up as rows silently missing a
    mass rather than as an error about the bad identifier.

    Args:
        term: The candidate.

    Returns:
        Whether it matches UniProtKB's accession syntax, which is `[OPQ][0-9][A-Z0-9]{3}[0-9]` or
        `[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]`, optionally with one further such group appended for the
        ten-character form.
    """
    return bool(re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", term))


def uniprot_batches(terms: Sequence[str], *, field: str, batch: int = 100) -> list[str]:
    """UniProt query strings covering every term, OR'd in batches.

    Args:
        terms: Accessions or entry names.
        field: The UniProt search field, `accession` or `id`.
        batch: Terms per query. Kept well below the service's 500-row page so one query's answer is
            never truncated by paging, which would silently lose accessions.

    Returns:
        One query string per batch, terms deduplicated and sorted so the same input gives the same
        queries and therefore the same cache entries.
    """
    unique = sorted({term for term in terms if term})
    return [
        " OR ".join(f"{field}:{term}" for term in unique[start : start + batch])
        for start in range(0, len(unique), max(batch, 1))
    ]


def parse_uniprot_tsv(text: str) -> list[Protein]:
    """A UniProt TSV response as records.

    Args:
        text: The TSV body, with the header row UniProt writes.

    Returns:
        The records. A row whose mass is not an integer is skipped: UniProt writes the mass with
        thousands separators in some responses and blank for a few entries, and a mass that cannot
        be read is a row the mass window cannot filter.
    """
    records = []
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        mass = (row.get("Mass") or "").replace(",", "").strip()
        if not mass.isdigit():
            continue
        records.append(
            Protein(
                accession=(row.get("Entry") or "").strip(),
                entry_name=(row.get("Entry Name") or "").strip(),
                protein_name=(row.get("Protein names") or "").strip(),
                organism=(row.get("Organism") or "").strip(),
                mass_da=int(mass),
            )
        )
    return records


def assign_accessions(
    pairs: Sequence[Pair],
    *,
    by_name: Mapping[str, str],
    by_chain: Mapping[tuple[str, str], str],
) -> tuple[list[tuple[Pair, str]], list[str]]:
    """Attach one UniProt accession to each pair, from whichever join can answer.

    A source carrying its own identifier is believed (Kincore, GPCRdb); SIFTS is for DynDom, which
    names only structures. When neither conformer has a hint both are resolved, and conformers that
    map to different proteins are rejected rather than coin-flipped.

    Args:
        pairs: The pairs.
        by_name: `{hint: accession}`, hints upper case.
        by_chain: `{(pdb, chain): accession}` from `sifts_chain_accessions`, with `""` accepted as a
            chain meaning "the entry's dominant accession".

    Returns:
        `(assigned, rejections)`, `assigned` being `(pair, accession)`.
    """
    assigned, rejections = [], []
    for pair in pairs:
        if pair.hint and pair.hint in by_name:
            assigned.append((pair, by_name[pair.hint]))
            continue
        first = _chain_accession(by_chain, pair.a)
        second = _chain_accession(by_chain, pair.b)
        label = f"{pair.source} {pair.name}"
        if first is None or second is None:
            missing = pair.a if first is None else pair.b
            hint = f" (hint {pair.hint} unresolved)" if pair.hint else ""
            rejections.append(f"{label}: no UniProt mapping for {missing.pdb}:{missing.chain or '*'}{hint}")
            continue
        if first != second:
            rejections.append(
                f"{label}: conformers map to different proteins, {pair.a.pdb}:{pair.a.chain or '*'} -> {first} "
                f"and {pair.b.pdb}:{pair.b.chain or '*'} -> {second}"
            )
            continue
        assigned.append((pair, first))
    return assigned, rejections


def _chain_accession(by_chain: Mapping[tuple[str, str], str], conformer: Conformer) -> str | None:
    """The accession of one conformer's chain, tolerating chain-id case.

    SIFTS reports the author chain id as deposited and DynDom's listing upper-cases some of them, so
    an exact-only lookup misses mappings that are present.

    Args:
        by_chain: The chain lookup.
        conformer: The conformer.

    Returns:
        The accession, or `None`.
    """
    for chain in (conformer.chain, conformer.chain.upper(), conformer.chain.lower(), ""):
        found = by_chain.get((conformer.pdb, chain))
        if found is not None:
            return found
    return None


def merge_by_accession(
    assigned: Sequence[tuple[Pair, str]], proteins: Mapping[str, Protein]
) -> tuple[list[SurveyRow], list[str]]:
    """One row per protein, carrying every pair every source offered for it.

    The deduplication the survey rests on. See the module docstring: a protein reached from two
    catalogs is one member of the population with two provenances, and counting it twice inflates
    the population exactly where the sources overlap.

    Args:
        assigned: `(pair, accession)` from `assign_accessions`.
        proteins: `{accession: protein}` from UniProt.

    Returns:
        `(rows, rejections)`, rows sorted by accession. An accession UniProt returned no record for
        is rejected with its pairs named -- a deleted or demerged accession is the usual cause, and
        it is a row that would otherwise have no mass and so silently escape the filter.
    """
    grouped: dict[str, list[Pair]] = {}
    for pair, accession in assigned:
        grouped.setdefault(accession, []).append(pair)

    rows, rejections = [], []
    for accession, pairs in sorted(grouped.items()):
        protein = proteins.get(accession)
        if protein is None:
            names = sorted({f"{pair.source}:{pair.name}" for pair in pairs})
            rejections.append(f"{accession}: no UniProt record, dropping {len(pairs)} pair(s) from {names}")
            continue
        rows.append(SurveyRow(protein=protein, pairs=tuple(sorted(pairs, key=lambda p: (p.source, p.motion, p.name)))))
    return rows, rejections


def summarize(rows: Sequence[SurveyRow]) -> dict[str, Any]:
    """Counts and spreads over a set of rows, as the numbers a plan document would quote.

    Args:
        rows: The rows.

    Returns:
        Row and pair counts by source and by motion class, the mass range, how many rows more than
        one source reached, and the rotation-angle spread over the pairs that have one.
    """
    pairs = [pair for row in rows for pair in row.pairs]
    masses = [row.protein.mass_kda for row in rows]
    angles = [pair.rotation_deg for pair in pairs if pair.rotation_deg is not None]
    return {
        "n_rows": len(rows),
        "n_pairs": len(pairs),
        "rows_by_source": _tally(source for row in rows for source in row.sources),
        "pairs_by_source": _tally(pair.source for pair in pairs),
        "pairs_by_motion": _tally(pair.motion for pair in pairs),
        "rows_by_n_sources": _tally(str(len(row.sources)) for row in rows),
        "mass_kda": _spread(masses),
        "rotation_deg": _spread(angles),
        "n_pairs_with_rotation": len(angles),
    }


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Counts per value, ordered by count and then by name so the output is stable.

    Args:
        values: The values.

    Returns:
        `{value: count}`.
    """
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    """Min, median and max of a set of numbers, or `None` when there are none.

    Args:
        values: The numbers.

    Returns:
        The three-number summary, rounded to three places -- these are reported, not computed on.
    """
    if not values:
        return None
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def scoreable_pairs(
    rows: Sequence[SurveyRow],
    *,
    require_angle: bool = True,
    allow_same_entry: bool = False,
    one_per_accession: bool = True,
) -> list[tuple[SurveyRow, Pair]]:
    """The pairs `panel/score_survey.py` can actually score, at one of two population widths.

    Both conformers must name a chain, always. The other three rules are switches, each a POPULATION
    decision rather than a feasibility one, and each measured before being made a default;
    docs/rationale/preprocess.md carries the reasoning.

    - `require_angle` decides whether the survey is one source or four. Only DynDom reports an angle.
    - `allow_same_entry` admits two chains of ONE deposition. Refused by default as "a crystal-packing
      comparison", though their Ca RMSD is indistinguishable from the accepted population, so the
      default is a stated choice rather than a finding.
    - `one_per_accession` is anti-pseudo-replication, and is a FITTING rule applied here for
      convenience. Applied at selection it is irreversible and its largest-angle tiebreak biases the
      population. Pass False to score every pair and group on the accession when fitting.

    Args:
        rows: Merged, mass-filtered rows, with DynDom details already attached.
        require_angle: Whether to demand a DynDom rotation angle.
        allow_same_entry: Whether to admit two chains of one PDB entry as a pair.
        one_per_accession: Whether to keep only the largest-angle pair per accession.

    Returns:
        `(row, pair)` in ascending order of angle. Angle-less pairs sort first, together, since they
        share a rank of -1. With `one_per_accession` false the pairs are deduplicated on
        `(pdb_a, chain_a, pdb_b, chain_b)`, two catalogs listing one pair being ordinary.
    """
    best: dict[str, tuple[SurveyRow, Pair]] = {}
    every: dict[tuple[str, str, str, str], list[tuple[SurveyRow, Pair]]] = {}
    for row in rows:
        for pair in row.pairs:
            if not (pair.a.pdb and pair.b.pdb and pair.a.chain and pair.b.chain):
                continue
            if not allow_same_entry and pair.a.pdb.lower() == pair.b.pdb.lower():
                continue
            if require_angle and pair.rotation_deg is None:
                continue
            if one_per_accession:
                prior = best.get(row.protein.accession)
                if prior is None or (pair.rotation_deg or -1.0) > (prior[1].rotation_deg or -1.0):
                    best[row.protein.accession] = (row, pair)
                continue
            # keyed on the CHAINS too: with `allow_same_entry` one deposition contributes several
            # distinct chain pairs, which a key of two PDB codes would collapse into one
            key = (pair.a.pdb.lower(), pair.a.chain, pair.b.pdb.lower(), pair.b.chain)
            every.setdefault(key, []).append((row, pair))
    if one_per_accession:
        return sorted(best.values(), key=lambda entry: entry[1].rotation_deg or -1.0)
    return sorted((_merge_duplicates(group) for group in every.values()), key=lambda e: e[1].rotation_deg or -1.0)


def _merge_duplicates(group: list[tuple[SurveyRow, Pair]]) -> tuple[SurveyRow, Pair]:
    """One scoring job per chain pair, carrying EVERY label the catalogs gave it.

    A pair listed twice is one measurement, so scoring it twice would double-weight it in any fit that
    did not group first. But the duplicates are not redundant: they are Kincore listing one pair under
    BOTH its label axes, so keeping whichever arrived first drops a class assignment. One row, then,
    with `extra["motions"]` carrying the set when it has more than one member -- the class analysis
    decides what an ambiguous row is worth instead of inheriting a dict's insertion order.

    Args:
        group: The `(row, pair)` entries sharing one `(pdb_a, chain_a, pdb_b, chain_b)`.

    Returns:
        The largest-angle entry, its pair carrying `motions` when the group disagreed.
    """
    row, pair = max(group, key=lambda entry: (entry[1].rotation_deg or -1.0, entry[1].motion))
    motions = sorted({other.motion for _, other in group})
    if len(motions) == 1:
        return row, pair
    return row, replace(pair, extra={**(pair.extra or {}), "motions": motions})


def stratify_by_angle(candidates: Sequence[tuple[SurveyRow, Pair]], *, n_wanted: int) -> list[tuple[SurveyRow, Pair]]:
    """A sample that spans the rotation-angle axis instead of piling up on its mode.

    Every k-th candidate after sorting by angle: the angle distribution is heavily skewed, so a
    uniform random sample of the same size would put most of its rows in one decile and leave the
    large-motion end to one or two. `d^2` is being plotted AGAINST this axis, so the sample has to
    cover it.

    Args:
        candidates: `scoreable_pairs` output, ascending in angle.
        n_wanted: How many rows to take. More than there are candidates returns all of them.

    Returns:
        The sample, ascending in angle, with no accession repeated.
    """
    if n_wanted >= len(candidates):
        return list(candidates)
    step = len(candidates) / n_wanted
    seen: set[str] = set()
    sample = []
    for index in range(n_wanted):
        row, pair = candidates[min(int(index * step), len(candidates) - 1)]
        if row.protein.accession not in seen:
            seen.add(row.protein.accession)
            sample.append((row, pair))
    return sample


_PANEL_NAME_LIMIT = 38
"""Characters of a protein name kept in a panel candidate's display name.

Cosmetic: the name is `"<protein> (<PDB_A>/<PDB_B>)"` and the pair of codes is what identifies the row.
"""


def panel_candidate(row: SurveyRow, pair: Pair) -> dict[str, Any]:
    """One survey pair as a `panel.json` candidate `reproduction/experiments/panel/score_survey.py` can read.

    Three settings are fixed by measured failures: `assembly` is `None` (the pair is defined on the
    named chains), `chains_a`/`chains_b` name one chain each, and `labels_a`/`labels_b` are both
    `["A"]` because two depositions name their chains independently and without relabeling the
    correspondence intersection is EMPTY. See docs/rationale/preprocess.md.

    Args:
        row: The protein.
        pair: The pair, which must satisfy `scoreable_pairs`.

    Returns:
        The candidate dict, carrying `rotation_deg`, `accession` and `source` as provenance that
        `targets.load_panel` ignores and a reader of the file does not.
    """
    pdb_a, pdb_b = pair.a.pdb.upper(), pair.b.pdb.upper()
    short = (row.protein.protein_name or row.protein.accession)[:_PANEL_NAME_LIMIT].strip()
    return {
        "name": f"{short} ({pdb_a}/{pdb_b})",
        "pdb_a": pdb_a,
        "pdb_b": pdb_b,
        "motion": pair.motion,
        # the angle is optional because only DynDom reports one; a Kincore or GPCRdb pair says so
        # rather than formatting a None, and the state labels are what stand in its place
        "note": (
            f"{row.protein.accession} {row.protein.mass_kda:.0f} kDa, {pair.source}, "
            + (
                f"rotation {pair.rotation_deg:.1f} deg"
                if pair.rotation_deg is not None
                else f"no rotation angle ({pair.source} reports none); states "
                f"{pair.a.state or '?'} -> {pair.b.state or '?'}"
            )
            + f", chains {pair.a.chain}/{pair.b.chain}"
        ),
        "assembly": None,
        "chains_a": [pair.a.chain],
        "chains_b": [pair.b.chain],
        "labels_a": ["A"],
        "labels_b": ["A"],
        "rotation_deg": pair.rotation_deg,
        "accession": row.protein.accession,
        "source": pair.source,
        # The mass the population was SCREENED on, not the mass the fit uses (`TargetRow.mass_kda`,
        # the matched atom set); the two diverge on a polyprotein. Carrying both lets a mass window be
        # revisited after scoring.
        "canonical_mass_kda": row.protein.mass_kda,
        # Two chains of ONE deposition, flagged rather than refused, so the crystal-packing question
        # is settled on the scored numbers. A field and not a filter on the name's PDB codes, which
        # would not survive a rename -- the coupling `--exclude-scored` documents having removed.
        "same_entry": pair.a.pdb.lower() == pair.b.pdb.lower(),
        # Only when the catalogs gave this one chain pair more than one class. `motion` above stays
        # one of them so the field never vanishes; a class analysis reads this. See `_merge_duplicates`.
        **({"motions": pair.extra["motions"]} if pair.extra and "motions" in pair.extra else {}),
        # Which naming `chains_a`/`chains_b` are written in, carried from the pair's own `extra`.
        # BioEmu names `label_asym_id` while selection downstream filters the AUTHOR chain; dropping
        # this would silently mispair the rows where the two namings disagree.
        "chain_id_kind": pair.extra.get("chain_id_kind", "auth_asym_id") if pair.extra else "auth_asym_id",
    }


SURVEY_ARMS: tuple[str, ...] = ("results/survey_pool/*.json",)
"""The surveyed population: the converged survey's per-pair scored pool.

Each `survey_pool/*.json` is one pair scored under the settled plan (`arguments` carry
`dose`/`pixel_size`/`box_size`/`rmsf`), which is what a driver reading only the plan needs. The two
retired predecessors -- the fixed-node `shift_survey_*`/`shift_wide_*` and the widened
`population_*` campaigns -- are superseded by this converged run and no longer on disk."""


POPULATION_ARMS: tuple[str, ...] = (
    "results/population_*.json",
    "results/populationfix_*.json",
    "results/populationkappa_*.json",
)
"""The widened survey, its three globs ORDERED rather than merely listed.

`results/population_*.json` is the campaign over the widened population. The two arms after it each
re-measure a subset of it, so a row can appear more than once and the LAST arm holding it wins:

* `results/populationfix_*.json` re-measures the rows whose marginal rung was lost when
  `measure_marginal` dropped `--min-shared-fraction` -- a configuration failure repaired by the
  corrected flag.
* `results/populationkappa_*.json` re-measures at more particles the rows the original refused for a
  NEGATIVE kappa -- the marginal rung falling between the two finest banks. A finite bank loses score
  rather than gaining it, so a fall means the gap between banks was smaller than the pool noise on it,
  and the only remedy is more particles. See `POPULATION_KAPPA_UNMEASURED` for the one not recovered.

Each repair was written BESIDE the arm it repairs, so the original campaign stays reproducible from
its own files.

**READ THESE WITH `supersede=True` AND IN THIS ORDER**, so the repairs win. Read strictly they raise
on the shared names (the guard working); read in the other order the stale null-marginal rows would
silently win. They are combinable because each repair reran under the original's own recorded
`arguments`, changing only what it had to: `--only` for both, and `--n-particles` for the kappa arm."""


POPULATION_KAPPA_UNMEASURED: dict[str, float] = {"Calmodulin-1 (1CLL/5GGM)": 53.99}
"""The one row still without a marginal, mapped to its measured LOWER BOUND (`d^2` at its finest bank).

Its two finest banks agree closely, which is why the extrapolation cannot fit: there is no resolvable
deficit left to add back, only noise. So this is not an unmeasured pair but a CONVERGED one that fails
an extrapolation guard written for unconverged ones. Quote it as `d^2_marginal >=` this value, the
finest bank actually evaluated, never as null -- a null reads as "we could not measure it", and the
truth is the opposite. It is excluded from fits regardless: a lower bound is not a point estimate, and
the census loses nothing by omitting one row."""


POPULATION_PANELS: tuple[str, ...] = ("data/survey_population_*.json",)
"""The panels behind `POPULATION_ARMS`.

One glob, not three, because both repair arms re-scored rows of these panels and wrote no panels of
their own. Note that a panel `name` is NOT unique here -- see `_pair_key` -- so a driver joining rows
to candidates must key on the chains as well, or it will match pairs to the wrong candidate."""


SURVEY_PANELS: tuple[str, ...] = (
    "data/survey_panel_*.json",
    "data/survey_pharma_*.json",
    "data/survey_nocontact*.json",
    "data/survey_wide_*.json",
)
"""The panel files behind `SURVEY_ARMS`, and the same trap one level up.

A panel carries the structures and chains a scored row was built from, so a driver that needs the
coordinates reads these while a driver that needs only the numbers reads `SURVEY_ARMS`.

**`data/survey_*.json` would be wrong for the same reason, and worse.** `data/survey_relax_*.json`
is a SUPERSET written by a later selection pass that re-emits every original and widened candidate.
Globbing `survey_*` therefore reads every pair twice, once from its own panel and once from the relax
copy, and no residual or `R^2` reveals it because the duplicate is an exact copy. Callers must refuse
a repeated pair name rather than trust the glob."""


def _pair_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """What identifies a surveyed pair, and why it is not the name.

    A row's `name` is a DISPLAY string, `"<protein> (<pdb_a>/<pdb_b>)"`, and it omits the chains. In
    the narrow population that was unique by accident, one pair per accession. The widened survey
    scores every chain pair, so some names cover two rows each -- different chain pairs, or the same
    entry at two rotations -- which are different measurements, and keying on the name alone would call
    them a double count and refuse the survey.

    `note` carries the chains and the rotation, and is generated rather than typed, so the pair is
    `(name, note)`, unique over the widened rows.

    Args:
        row: A `score_survey.py` result row.

    Returns:
        The key, or `None` for a row carrying no name, which cannot be checked at all.
    """
    name = row.get("name")
    if name is None:
        return None
    return (name, row.get("note"))


_CHAINS_IN_NOTE = re.compile(r"chains ([^/,]+)/([^/,]+)\s*$")
"""Where a row records which chains it scored. `note` is generated, not typed."""


def join_key(item: Mapping[str, Any] | Any) -> tuple[Any, ...]:
    """How to join a scored row to the panel candidate it came from.

    NOT the name. A name omits the chains, and the widened survey scores every chain pair, so some
    names cover two rows each -- joining on the name alone silently pairs one row's `d^2` with another
    row's geometry, which no residual reveals because both halves are real numbers.

    Accepts either side: a result row, whose chains are read out of `note`, or a `Candidate`, whose
    chains are fields.

    Args:
        item: A `score_survey.py` result row, or a `reproduction.targets.Candidate`.

    Returns:
        `(name, chains_a, chains_b)`, with the chains as comma-joined strings.
    """
    if isinstance(item, Mapping):
        found = _CHAINS_IN_NOTE.search(str(item.get("note") or ""))
        if found is None:
            return (item.get("name"), None, None)
        return (item.get("name"), found.group(1).strip(), found.group(2).strip())
    return (item.name, ",".join(item.chains_a or ()), ",".join(item.chains_b or ()))


def read_survey_rows(patterns: str | Sequence[str] = SURVEY_ARMS, *, supersede: bool = False) -> list[dict[str, Any]]:
    """Every surveyed pair, from one or more globs over `score_survey.py` records.

    Args:
        patterns: A glob or a sequence of them. Defaults to `SURVEY_ARMS`, both campaigns.
        supersede: Whether a LATER pattern's row replaces an earlier pattern's row of the same
            name, in place. Off by default, because a repeated pair is normally a double count.
            Turn it on only where one arm deliberately re-measures rows of another, as
            `POPULATION_ARMS` does -- and note that pattern ORDER then carries the precedence, so
            passing the arms the other way round silently keeps the superseded rows. A repeat
            within a single pattern still raises, on or off.

    Returns:
        The rows, in file then record order. Callers apply their own filters -- one figure needs a
        positive marginal rung and another needs every row -- so nothing is dropped here.

    Raises:
        SystemExit: If the patterns match no file, if they match no row, or if two rows carry the
            same non-null `name` under `supersede=False`, or within one pattern under either. The
            last is the one worth having: a repeated pair is a double-counted measurement, and a fit
            over double-counted rows converges to a plausible wrong answer rather than failing. Rows
            carrying no name pass through unchecked.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    paths: list[tuple[int, Path]] = []
    seen_paths: set[Path] = set()
    for rank, pattern in enumerate(patterns):
        for name in sorted(glob(pattern)):
            resolved = Path(name).resolve()
            # the same file reached through two patterns is a caller's typo, not a double count
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                paths.append((rank, Path(name)))
    if not paths:
        raise SystemExit(f"{list(patterns)} matches no files")

    rows: list[dict[str, Any]] = []
    origin: dict[str, tuple[int, str]] = {}
    where: dict[str, int] = {}
    for rank, path in paths:
        result = read_payload(path)
        for row in result.get("rows", []):
            name = _pair_key(row)
            # A row without a name cannot be checked, and must not be treated as a duplicate of the
            # next nameless row -- every real survey row carries one, but a fixture need not.
            if name is None:
                rows.append(row)
                continue
            if name in origin:
                seen_rank, seen_path = origin[name]
                # A repeat within one pattern is always a double count, whatever `supersede` says:
                # one arm cannot re-measure itself, so it is a glob typo or a duplicated row.
                if not supersede or seen_rank == rank:
                    raise SystemExit(
                        f"pair {name[0]!r} ({name[1]!r}) appears in both {seen_path} and {path}; the "
                        f"patterns {list(patterns)} overlap and would double-count it"
                    )
                rows[where[name]] = row
                origin[name] = (rank, str(path))
                continue
            origin[name] = (rank, str(path))
            where[name] = len(rows)
            rows.append(row)
    if not rows:
        raise SystemExit(f"{list(patterns)} matched {len(paths)} files but no row")
    return rows
