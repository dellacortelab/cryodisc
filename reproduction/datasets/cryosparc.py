"""What a preprocessing chain writes down, and how to read what cryoSPARC left on disk.

The schema three drivers share. A record carries the JOB DIRECTORY of every stage, and files are
derived from that directory rather than from a retyped name. Reading needs only numpy: a `.cs` is a
plain `.npy` holding a structured array, so nothing here talks to a cryoSPARC server or needs
`cryosparc-tools`. Pools are written as `.cs` too, so `score_pool.py --meta` reads them with no
second reader to drift against.

Rationale: docs/rationale/preprocess.md
"""

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from cryodisc._optional import require_mrcfile
from cryodisc.particles.metadata import read_cs_array

STAGES = ("import", "motion", "ctf", "pick", "extract", "class2D")
"""The chain, in order. `cryosparc_chain.py --stop-after` indexes into this."""

ALIGNED_STAGES = ("import", "ctf", "pick", "extract", "class2D")
"""The chain for an entry that deposits ALIGNED micrographs, which has no motion-correction step.

When an entry deposits patch-aligned dose-weighted micrographs, motion correction and dose weighting
are the depositors' and not ours. `--stop-after motion` cannot be honored on this route, so it
indexes a different tuple rather than pretending the stage exists.
"""


def stages_for(aligned: bool) -> tuple[str, ...]:
    """Which stage tuple a route runs, chosen in one place. See docs/rationale/preprocess.md.

    Args:
        aligned: Whether the deposition ships aligned micrographs rather than movies.

    Returns:
        The stages, in order.
    """
    return ALIGNED_STAGES if aligned else STAGES


@dataclass(frozen=True)
class Stage:
    """One finished cryoSPARC job.

    Args:
        job: Job uid, e.g. `J14`.
        type: cryoSPARC job type, e.g. `patch_ctf_estimation_multi`.
        status: The job's terminal status. Anything but `completed` means its outputs are not there.
        num_items: Items across the job's output groups, which for a pick or extract is particles.
        directory: Absolute path of the job directory, which is where every file below is found.
        outputs: Output group names, recorded because they drift between cryoSPARC versions.
    """

    job: str
    type: str
    status: str
    num_items: int
    directory: str
    outputs: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        """Whether the job finished, which is the only state whose outputs may be read."""
        return self.status == "completed"


@dataclass(frozen=True)
class RunRecord:
    """One imageset's chain: what was run, on what, and where its outputs are.

    Args:
        label: The state this imageset stands for -- the benchmark's ground truth, assigned at launch.
        entry: The accession the movies came from.
        imageset: The imageset directory within the entry, kept separate from `entry` because two
            sessions of one entry are same-state pools the cross-session null needs kept apart.
        project: cryoSPARC project uid.
        project_dir: The project directory `blob/path` is relative to; what `score_pool.py --root` wants.
        workspace: cryoSPARC workspace uid.
        picker: `blob` or `template`.
        template_volume: The MRC the templates were built from, or `None` for blob picking.
        movies: The import glob.
        stages: Finished jobs by stage name; see `STAGES`.
        source: The file this was read from, or `""` for one just built.
    """

    label: str
    entry: str
    imageset: str
    project: str
    project_dir: str
    workspace: str
    picker: str = "blob"
    template_volume: str | None = None
    movies: str = ""
    stages: dict[str, Stage] = field(default_factory=dict)
    source: str = ""

    @property
    def tag(self) -> str:
        """A pool name carrying the state AND the session.

        The session, not just the state: the cross-session same-state null needs the sessions kept
        apart, and the recovered fraction varies between sessions of one entry by more than the two
        states vary from each other.
        """
        return f"{self.label}_{self.entry}_{self.imageset}"

    def completed_stage(self, name: str) -> Stage:
        """The named stage, or a refusal naming what the record actually holds.

        Args:
            name: A member of `STAGES`.

        Returns:
            The stage.

        Raises:
            KeyError: If the stage is absent or did not complete. Raised rather than returning
                `None`, because the caller's next move is to open a file in a directory that a
                failed job did not fill.
        """
        stage = self.stages.get(name)
        if stage is None:
            raise KeyError(f"{self.source or self.tag}: no {name} stage; it has {sorted(self.stages)}")
        if not stage.completed:
            raise KeyError(f"{self.source or self.tag}: {name} job {stage.job} is {stage.status!r}, not completed")
        return stage

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON, `source` excluded -- it is where the file is, not what it says."""
        return {
            "label": self.label,
            "entry": self.entry,
            "imageset": self.imageset,
            "project": self.project,
            "project_dir": self.project_dir,
            "workspace": self.workspace,
            "picker": self.picker,
            "template_volume": self.template_volume,
            "movies": self.movies,
            "stages": {name: asdict(stage) | {"outputs": list(stage.outputs)} for name, stage in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, stored: dict[str, Any], source: str = "") -> "RunRecord":
        """Rebuild a record read from JSON.

        Args:
            stored: The dict `to_dict` wrote.
            source: The file it came from.

        Returns:
            The record.

        Raises:
            ValueError: If the dict is not a run record. A run directory also collects parameter
                dumps and other bookkeeping, so a caller's glob is not evidence of content, and
                skipping a stray file quietly is worse than crashing on it.
        """
        required = {"label", "entry", "imageset", "stages"}
        if not required <= set(stored):
            raise ValueError(f"{source or stored!r}: not a run record; it is missing {sorted(required - set(stored))}")
        return cls(
            label=str(stored["label"]),
            entry=str(stored["entry"]),
            imageset=str(stored["imageset"]),
            project=str(stored.get("project", "")),
            project_dir=str(stored.get("project_dir", "")),
            workspace=str(stored.get("workspace", "")),
            picker=str(stored.get("picker", "blob")),
            template_volume=stored.get("template_volume"),
            movies=str(stored.get("movies", "")),
            stages={
                name: Stage(
                    job=str(value["job"]),
                    type=str(value.get("type", "")),
                    status=str(value.get("status", "")),
                    num_items=int(value.get("num_items", 0)),
                    directory=str(value.get("directory", "")),
                    outputs=tuple(value.get("outputs", ())),
                )
                for name, value in stored["stages"].items()
            },
            source=source,
        )


def write_run_record(path: str | Path, record: RunRecord) -> Path:
    """Write a run record, creating its directory.

    Meant to be written after every stage, so a walltime expiring mid-chain does not lose the uids
    of the jobs that did finish.

    Args:
        path: Destination JSON.
        record: The record.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return destination


def read_run_record(path: str | Path) -> RunRecord:
    """Read a run record.

    Args:
        path: The JSON `write_run_record` produced.

    Returns:
        The record.

    Raises:
        ValueError: If the file is not a run record.
    """
    return RunRecord.from_dict(json.loads(Path(path).read_text()), source=str(path))


def read_run_records(
    patterns: Iterable[str | Path], stage: str
) -> tuple[list[tuple[RunRecord, Stage]], list[dict[str, str]]]:
    """Every run record a glob matched that reached a stage, and the reason for each one left out.

    A record is identified by its CONTENT and not by the caller's glob, so bookkeeping a run directory
    also collects is left out rather than crashed on -- and left out with a REASON, never quietly. See
    docs/rationale/preprocess.md.

    Args:
        patterns: Glob patterns, `~` expanded. Matches are deduplicated and sorted, so overlapping
            globs cannot present one record twice.
        stage: The stage each record must have completed; see `STAGES`.

    Returns:
        `(found, skipped)`. `found` is `(record, that stage)` per usable record in path order;
        `skipped` is `{"run", "why"}` per file left out, which is the shape a result JSON wants.
    """
    found: list[tuple[RunRecord, Stage]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted({match for pattern in patterns for match in glob(str(Path(pattern).expanduser()))}):
        try:
            record = read_run_record(path)
            completed = record.completed_stage(stage)
        except (ValueError, KeyError) as reason:
            skipped.append({"run": path, "why": str(reason)})
            continue
        found.append((record, completed))
    return found, skipped


def job_file(job_dir: str | Path, suffix: str) -> Path:
    """The newest file in a job directory ending in `suffix`, rejects and passthroughs excluded.

    cryoSPARC names outputs after the job and the iteration, so a directory holds several generations
    of the same file and the last is the result. Rejects and passthroughs match the suffix and are not.

    Args:
        job_dir: The job directory.
        suffix: What the file ends with, e.g. `particles.cs`.

    Returns:
        The path.

    Raises:
        FileNotFoundError: If nothing matches, naming the directory searched.
    """
    directory = Path(job_dir)
    found = sorted(
        path for path in directory.glob(f"*{suffix}") if "passthrough" not in path.name and "rejected" not in path.name
    )
    if not found:
        raise FileNotFoundError(f"{directory}: no *{suffix}; is this the job directory the record recorded?")
    return found[-1]


def credentials_path() -> Path:
    """Where the three-line cryoSPARC credentials file lives, honoring `CRYOSPARC_LICENSE_FILE`.

    Resolved here rather than defaulted in each driver's `argparse`, so one override reaches every
    driver and a hardcoded fallback cannot silently ignore an operator who overrode it.

    Returns:
        The path, `~` expanded but not required to exist.
    """
    return Path(os.environ.get("CRYOSPARC_LICENSE_FILE") or "~/.cryosparc/license_id").expanduser()


def connect(host: str, license_file: Path | None = None, port: int = 39000) -> tuple:
    """A cryoSPARC session from a license file holding license, email and password on three lines.

    Args:
        host: cryoSPARC master FQDN.
        license_file: The three-line credentials file. `None` takes `credentials_path()`, so one
            override reaches every driver.
        port: The master's base port, from the caller because `CRYOSPARC_BASE_PORT` is overridable.

    Returns:
        `(session, email)`. The email is returned because creating a project needs the user id it
        resolves to.

    Raises:
        SystemExit: If the connection fails.
    """
    from cryosparc.tools import CryoSPARC

    path = credentials_path() if license_file is None else Path(license_file).expanduser()
    # three lines: license, email, password
    license_id, email, password = (line.strip() for line in path.read_text().splitlines()[:3])
    session = CryoSPARC(license=license_id, host=host, base_port=int(port), email=email, password=password)
    if not session.test_connection():
        raise SystemExit(f"could not reach the cryoSPARC master at {host}:{port}")
    return session, email


def output_named(job, *wanted: str) -> str:
    """The job's output group matching any of `wanted`, or a failure listing what it has.

    cryoSPARC's output group names drift between versions, so a chain that hardcodes one breaks on
    upgrade in the middle of a queued run rather than at the call.

    Args:
        job: The cryoSPARC job.
        *wanted: Substrings to look for, in order of preference.

    Returns:
        The output group name.

    Raises:
        SystemExit: If nothing matches.
    """
    names = [group.get("name") for group in job.doc.get("output_result_groups", [])]
    for want in wanted:
        hit = next((name for name in names if want in (name or "").lower()), None)
        if hit:
            return hit
    raise SystemExit(f"{job.uid} has no output matching {wanted}; it has {names}")


def read_particles(job_dir: str | Path) -> np.ndarray:
    """Every particle a job output, with its passthrough joined on.

    The join is the point: a 2D classification writes `alignments2D/*` and passes `blob/*` and `ctf/*`
    through, so the particles file alone names particles that cannot be scored.

    Args:
        job_dir: The job directory.

    Returns:
        The structured array, one row per particle.

    Raises:
        FileNotFoundError: If the directory holds no particles file.
        ValueError: If a passthrough is present but does not cover the same particles.
    """
    directory = Path(job_dir)
    particles = job_file(directory, "particles.cs")
    passthrough = sorted(directory.glob("*passthrough*particles.cs"))
    return read_cs_array(particles, passthrough=passthrough[-1] if passthrough else None)


def write_particles(path: str | Path, data: np.ndarray) -> Path:
    """Write a structured array as a `.cs` that `particles.metadata.load` can read.

    Through an open file handle rather than by name, because `np.save` appends `.npy` to any path that
    does not already end in it -- `pool.cs` would land as `pool.cs.npy`.

    Args:
        path: Destination, conventionally `.cs`.
        data: The structured array.

    Returns:
        The path written.

    Raises:
        ValueError: If there are no particles. An empty pool is a file that looks like a result and
            scores as nothing; the caller has a reason for the emptiness and should say it.
    """
    if len(data) == 0:
        raise ValueError(f"{path}: refusing to write a pool of zero particles")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        np.save(handle, data, allow_pickle=False)
    return destination


def read_class_averages(job_dir: str | Path) -> tuple[np.ndarray, float]:
    """The 2D class averages a job produced, and the pixel size they are on.

    The pixel size comes from the averages' own `.cs` and not the extraction settings: a classification
    may downsample, and a diameter converted with the wrong pixel size is wrong by that ratio, silently.

    Args:
        job_dir: A `class_2D` job directory.

    Returns:
        `(images, pixel_size)` with `images` as `(k, y, x)` float32.

    Raises:
        FileNotFoundError: If the directory holds no class averages.
        ImportError: If `mrcfile` is not installed.
        ValueError: If the `.cs` beside them holds no rows or no pixel size, which is what an aborted
            classification leaves behind.
    """
    mrcfile = require_mrcfile("reading class averages")
    directory = Path(job_dir)
    with mrcfile.open(job_file(directory, "class_averages.mrc"), permissive=True) as handle:
        images = np.asarray(handle.data, np.float32)
    header_path = job_file(directory, "class_averages.cs")
    header = np.load(header_path, allow_pickle=False)
    # named rather than left as a bare IndexError/KeyError, so an aborted classification's leftover
    # `class_averages.cs` gives an answer about the job directory, not about which numpy line failed
    if "blob/psize_A" not in (header.dtype.names or ()):
        raise ValueError(f"{header_path} carries no blob/psize_A; this is not a class_averages table")
    if len(header) == 0:
        raise ValueError(f"{header_path} holds no rows, so the job wrote no class averages to size")
    return images, float(np.asarray(header["blob/psize_A"])[0])


def extraction_grid(data: np.ndarray) -> tuple[int, float]:
    """The box and pixel size the particles' IMAGES are stored on.

    Counterpart of `read_class_averages`' pixel size, not a substitute: a classification may
    downsample, and the two grids then differ by exactly that ratio.

    Args:
        data: Particles carrying `blob/*`: an extract job's output, or a classification's with its
            passthrough joined in by `read_particles`.

    Returns:
        `(box_size, pixel_size)`, the stored box edge in pixels and its A/px.

    Raises:
        KeyError: If the particles carry no `blob/*` (a classification's own file read without its
            passthrough).
        ValueError: If the set is empty, or its rows disagree (it spans two extraction grids).
    """
    fields = data.dtype.names or ()
    missing = [name for name in ("blob/shape", "blob/psize_A") if name not in fields]
    if missing:
        raise KeyError(f"these particles carry no {missing}; a class_2D job passes blob/* through, so join it")
    if not len(data):
        raise ValueError("an empty particle set has no extraction grid")
    shapes = np.unique(np.asarray(data["blob/shape"], int), axis=0)
    pixels = np.unique(np.asarray(data["blob/psize_A"], float))
    if len(shapes) > 1 or len(pixels) > 1:
        raise ValueError(
            f"these particles span more than one extraction grid: boxes {shapes.tolist()} at "
            f"{pixels.tolist()} A/px. One box and one pixel size cannot describe them"
        )
    return int(shapes[0][0]), float(pixels[0])


def micrograph_grid(data: np.ndarray) -> float:
    """The pixel size the particles' MICROGRAPHS are on, which extraction usually downsamples away.

    `extraction_grid`'s counterpart, and what anything fitting a micrograph needs: using the boxes'
    downsampled pixel size instead propagates through `chi = pi lambda dz k^2` into the fitted
    defocus. Only the pixel size, not the shape: one detector and magnification give one pixel size
    but can still give two shapes.

    Args:
        data: Particles carrying `location/micrograph_psize_A`: a pick job's output, or a later
            job's with its passthrough joined in by `read_particles`.

    Returns:
        The micrograph pixel size in A/px.

    Raises:
        KeyError: If the particles carry no `location/*` (a `.star`, or a `.cs` read without its
            passthrough).
        ValueError: If the set is empty, or spans more than one micrograph sampling.
    """
    if "location/micrograph_psize_A" not in (data.dtype.names or ()):
        raise KeyError(
            "these particles carry no location/micrograph_psize_A; it comes from a pick job and "
            "rides through a class_2D job's passthrough, so join it"
        )
    if not len(data):
        raise ValueError("an empty particle set has no micrograph grid")
    pixels = np.unique(np.asarray(data["location/micrograph_psize_A"], float))
    if len(pixels) > 1:
        raise ValueError(
            f"these particles span more than one micrograph sampling: {pixels.tolist()} A/px. "
            "One pixel size cannot describe them"
        )
    return float(pixels[0])


def class_labels(data: np.ndarray) -> np.ndarray:
    """The 2D class each particle landed in.

    Args:
        data: Particles from a `class_2D` job.

    Returns:
        (n,) integer class indices.

    Raises:
        KeyError: If the set was never classified, which is the likeliest way a wrong job directory
            reaches this far.
    """
    if "alignments2D/class" not in (data.dtype.names or ()):
        raise KeyError("these particles carry no alignments2D/class; a class_2D job is what writes it")
    return np.asarray(data["alignments2D/class"], int)


def class_counts(data: np.ndarray, n_classes: int) -> np.ndarray:
    """Particles per 2D class, including the classes that took none.

    Args:
        data: Particles from a `class_2D` job.
        n_classes: How many classes the job produced, from the class averages not the labels -- an
            empty class is the one a `bincount` over labels would drop, shifting every later index.

    Returns:
        (n_classes,) counts, summing to the particle count.

    Raises:
        ValueError: If a particle carries a class index `n_classes` does not cover -- a job directory
            whose `class_averages.mrc` does not belong to its particles.
    """
    labels = class_labels(data)
    counts = np.bincount(labels, minlength=n_classes)
    if counts.size > n_classes:
        raise ValueError(
            f"a particle names class {int(labels.max())} but only {n_classes} class averages were "
            "read. The averages are what say how many classes there are, so this is a job directory "
            "whose class_averages.mrc does not belong to its particles -- check the record's job uid."
        )
    return counts


def select_classes(data: np.ndarray, classes: list[int] | np.ndarray) -> np.ndarray:
    """The particles of the named 2D classes.

    Args:
        data: Particles from a `class_2D` job.
        classes: Class indices to keep.

    Returns:
        The subset, in the original row order.
    """
    return data[np.isin(class_labels(data), np.asarray(classes, int))]


def class_resolution(job_dir: str | Path) -> np.ndarray:
    """Per-class estimated resolution (A) from a class_2D job's class averages.

    Args:
        job_dir: A `class_2D` job directory.

    Returns:
        (k,) resolutions in A, one per class average, aligned with `class_counts`' indices.

    Raises:
        ValueError: If the `.cs` beside the averages carries no `blob/res_A` -- not a class_2D output.
    """
    header = np.load(job_file(Path(job_dir), "class_averages.cs"), allow_pickle=False)
    if "blob/res_A" not in (header.dtype.names or ()):
        raise ValueError(f"{job_dir}: class_averages.cs carries no blob/res_A; this is not a class_2D output")
    return np.asarray(header["blob/res_A"], float)


def resolved_classes(
    res_A: np.ndarray,
    counts: np.ndarray,
    *,
    res_target: float = 15.0,
    res_ceiling: float = 25.0,
    min_particles: int = 25000,
) -> np.ndarray:
    """The 2D classes to keep: the well-resolved ones, relaxed only as far as a count margin needs.

    Junk removal on per-class RESOLUTION, never on state, so it leaves non-circularity intact: ice,
    carbon and empty classes resolve poorly whichever conformer a grid holds. Achievable 2D resolution
    is system-dependent, so the cut adapts -- every class at least as good as `res_target` is kept, and
    while that holds fewer than `min_particles` the next-best classes are admitted in resolution order,
    but never one worse than `res_ceiling` (a floor that admits no genuine junk to reach a count).

    Args:
        res_A: Per-class resolution, from `class_resolution`.
        counts: Per-class particle counts, from `class_counts`; empty classes are ignored.
        res_target: Keep every occupied class at least this well-resolved.
        res_ceiling: Never keep a class worse than this, even to reach `min_particles`.
        min_particles: Relax past `res_target` until at least this many particles are kept.

    Returns:
        Sorted class indices to keep.

    Raises:
        ValueError: If `res_A` and `counts` are not one-per-class.
    """
    res_A = np.asarray(res_A, float)
    counts = np.asarray(counts, int)
    if res_A.shape != counts.shape:
        raise ValueError(f"{res_A.shape} resolutions but {counts.shape} counts; one of each per class is needed")
    kept: list[int] = []
    total = 0
    for k in sorted((i for i in range(len(res_A)) if counts[i] > 0), key=lambda i: res_A[i]):
        if res_A[k] <= res_target or (total < min_particles and res_A[k] <= res_ceiling):
            kept.append(k)
            total += int(counts[k])
    return np.array(sorted(kept), int)
