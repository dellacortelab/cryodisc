"""Take a real particle set from its metadata file to a per-particle log-ratio.
The whole real-data arm assembled: read the table, open the stacks, group the optics, calibrate. The
`.npy` written here is what `benchmark/known_fraction.py --log-ratio-a/-b` reads.
    .venv/bin/python .../score_pool.py --meta J42/particles.cs --root /projects/CS-run --structure-a a.pdb --structure-b b.pdb --out out.json --log-ratio lambda.npy
Three flags exist because real data can be wrong where simulated cannot: `--pixel-size` overrides a
deposited pixel size, `--refit-ctf` refits a deposited defocus (at `--micrograph-pixel-size`), and
`--check-shift-sign` measures the recentring sign. `--calibrate poses` (default under `--known-pose`)
fits the contrast at each particle's own pose; `bank` biases it low. Never read a `purity`/`sweep` off
a suspect contrast, and treat `separation` above one as an alarm (the sweep is not scale-invariant).
"""

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cryodisc import CollectionPlan
from cryodisc.grids.fourier import shell_indices
from cryodisc.grids.poses import (
    euler_from_relion,
    expansion_plan_or_none,
    pose_bank,
    realized_nodes,
    shift_grid,
)
from cryodisc.particles.em import calibrate_at_known_poses, fit_noise_model
from cryodisc.particles.images import read_micrograph
from cryodisc.particles.metadata import (
    band_limit_envelope,
    read_cs_array,
    transfer_from_meta,
)
from cryodisc.particles.snr import matched_filter_snr
from cryodisc.pools import build_states as pool_build
from cryodisc.pools import read_pool as pool_read
from cryodisc.rungs.marginal import (
    known_pose_d2,
    known_pose_log_ratio,
    pose_log_ratio,
    streamed_pose_log_ratio,
)
from cryodisc.statistics.convergence import converge_marginal
from cryodisc.statistics.information import deflection
from reproduction.operating_point import SHARED
from reproduction.record import write_result
from reproduction.targets import add_assembly_argument
from reproduction.validation.coherence import coherence, fractional_error, predicted_coherence
from reproduction.validation.floor import shuffle_floor

DRIVER = "reproduction/experiments/realdata/score_pool.py"

_DEFAULT_NODES = 54
"""Pose bank when marginalizing. Deliberately absent under `--known-pose`; see `parse_args`."""


def _fitted_shift_grid(box_size: int, args: argparse.Namespace, shift_sigma) -> object:
    """The shift grid at the EM-fitted Gaussian width, so scoring marginalizes under that prior.

    The grid the calibration was handed carries a uniform prior; the width it fitted replaces it here.

    Args:
        box_size: Image size the ramps are built on.
        args: Parsed arguments, for the window and step.
        shift_sigma: The fitted Gaussian width in pixels, floored off zero.

    Returns:
        A `ShiftGrid` on the same lattice, carrying the fitted Gaussian log-weights.
    """
    return shift_grid(box_size, float(args.shift_window), sigma=max(float(shift_sigma), 1e-6), step=args.shift_step)


def _marginal_scores(models, grid, images, noise_spectrum, contrast, shifts, transfer, box_size):
    """Pose-marginalized Lambda over `grid`, rendered and reduced in CHUNKS so a large bank fits memory.

    `pose_log_ratio` materializes the whole bank, which overflows at a converged bank on a large box;
    `streamed_pose_log_ratio` renders it in pose chunks and builds 7 of every 8 templates from the
    expansion plan, so the bank can grow to convergence. The answer is identical.

    Args:
        models: The two `CryoJAXModel`s, whose `bare_contrast` renders each state's templates.
        grid: The pose bank to marginalize over.
        images: fft2 of the particle stack.
        noise_spectrum: Fitted sigma^2(k).
        contrast: Fitted template scale.
        shifts: Translation lattice, or `None` for the fast centered path.
        transfer: Per-group transfer.
        box_size: Image edge, for the expansion plan.

    Returns:
        Per-particle marginal Lambda.
    """
    return streamed_pose_log_ratio(
        images,
        models[0].bare_contrast,
        models[1].bare_contrast,
        grid.angles,
        grid.log_weights,
        noise_spectrum,
        amplitude=contrast,
        shifts=shifts,
        transfer=transfer,
        expansion=expansion_plan_or_none(grid, box_size),
    ).marginal


def read_pool(args: argparse.Namespace):
    """`pool.read_pool` with this driver's flag names.

    Args:
        args: Parsed arguments.

    Returns:
        `(meta, images, pixel_size, shift_report, labels)`.
    """
    return pool_read(
        args.meta,
        root=args.root,
        states=args.state,
        limit=args.limit or 0,
        subsample=args.subsample or 0,
        subsample_seed=args.seed,
        box_size=args.box_size,
        pixel_size=args.pixel_size,
        cs_pose_transpose=None if args.cs_pose_transpose is None else args.cs_pose_transpose == "yes",
        invert_contrast=args.invert_contrast,
        recenter_images=not args.no_recenter,
        check_shift_sign=args.check_shift_sign,
    )


def build_states(args: argparse.Namespace, plan: CollectionPlan, box_size: int, pixel_size: float):
    """`pool.build_states` with this driver's flag names.

    Args:
        args: Parsed arguments.
        plan: The collection.
        box_size: Image size the particles were extracted at.
        pixel_size: Angstroms per pixel.

    Returns:
        `(models, report)`.
    """
    return pool_build(
        plan,
        box_size=box_size,
        pixel_size=pixel_size,
        structure_a=args.structure_a,
        structure_b=args.structure_b,
        map_a=args.map_a,
        map_b=args.map_b,
        assembly=args.assembly,
        # this driver declares no `--chains`; `check_pool_arguments` documents the case
        chains=None,
        ligands=tuple(args.ligands or ()),
        superpose=not args.no_superpose,
        offset_from_maps=args.structure_offset_from_maps,
        ignore_origin=args.ignore_origin,
    )


def refit_ctf(meta, args: argparse.Namespace):
    """Refit the defocus from micrographs and splice it in, or leave the table as deposited.

    Args:
        meta: The metadata table.
        args: Parsed arguments.

    Returns:
        `(meta, report)`. The report records what MOVED, since a refit that changed nothing and a
        refit that never ran are different findings and only one of them is reassuring.
    """
    if not args.refit_ctf:
        return meta, {"ran": False}

    from cryodisc.particles.ctffit import fit_patches, interpolate_to_particles

    # the optics off the table that knows them (`fit_patches` would default to a wrong wavelength), at
    # the MICROGRAPH's pixel size, since `chi = pi lambda dz k^2` turns a factor error in k into dz^2
    optics = meta.optics(pixel_size=args.micrograph_pixel_size)
    patches = fit_patches(read_micrograph(args.micrograph), **optics.as_kwargs())
    # the PATCH CENTROID, not the origin: `ParticleMeta` carries no micrograph coordinates, so every
    # particle takes the mean point the tilted defocus plane was fitted about, not a corner extrapolation
    centroid = np.array([[np.mean([p[0] for p in patches]), np.mean([p[1] for p in patches])]])
    # spliced in as fitted, with no grid conversion: the defocus is a physical length in angstroms and
    # carries no dependence on the grid it was measured on
    fitted = interpolate_to_particles(patches, np.repeat(centroid, len(meta), axis=0))
    before = float(np.median(meta.defocus))
    meta = meta.with_ctf(
        defocus_u=fitted["defocus_u"],
        defocus_v=fitted["defocus_v"],
        astigmatism_angle=fitted["astigmatism_angle"],
    )
    after = float(np.median(meta.defocus))
    return meta, {
        "ran": True,
        "micrograph_pixel_size": float(args.micrograph_pixel_size),
        "median_defocus_before": before,
        "median_defocus_after": after,
        "ratio": after / before,
    }


def stated_micrograph_pixel_size(meta: Path) -> float | None:
    """The micrograph A/px the metadata file itself carries, or `None` when it carries none.

    Args:
        meta: The `--meta` file.

    Returns:
        The pixel size, or `None` for a `.star` and for a `.cs` with no `location/*` -- neither is an
        error, since the flag exists to supply exactly that case.

    Raises:
        ValueError: If the file states more than one, which no single fit can serve.
    """
    if Path(meta).suffix.lower() != ".cs":
        return None
    from reproduction.datasets.cryosparc import micrograph_grid

    try:
        return micrograph_grid(read_cs_array(meta))
    except KeyError:
        return None


def check_pool_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse an argument combination that names no two states, or names them twice.

    `pools.build_states` raises on the same combinations; this buys a usage message and exit 2 in
    place of a traceback, plus the argparse-only checks below that `build_states` cannot make.

    Args:
        parser: The parser, for `parser.error` -- a usage message and exit 2, not a traceback.
        args: What it parsed.
    """
    # neither route is defaulted: scoring against the wrong state's template reads as a smaller
    # signal rather than failing
    structures = args.structure_a is not None or args.structure_b is not None
    maps = args.map_a is not None or args.map_b is not None
    if structures and maps:
        parser.error("pass either --structure-a/-b or --map-a/-b, not both: they name the same two states")
    if not structures and not maps:
        parser.error("need two states: --structure-a/-b for atomic models, or --map-a/-b for voxel maps")
    if structures and not (args.structure_a and args.structure_b):
        parser.error("--structure-a and --structure-b are both required: a log-ratio takes two states")
    if maps and not (args.map_a and args.map_b):
        parser.error("--map-a and --map-b are both required: a log-ratio takes two states")
    # a half-map carries reconstruction noise the oracle counts as signal: it inflates d^2 and widens
    # the score variance, so d^2_obs reads low. Refuse unless the caller says they mean it.
    if maps and not args.allow_half_map:
        for flag, path in (("--map-a", args.map_a), ("--map-b", args.map_b)):
            if path and ("_half_A" in Path(path).name or "_half_B" in Path(path).name):
                parser.error(
                    f"{flag}={Path(path).name} looks like a half-map, whose noise the oracle counts as "
                    "signal. Build a full template with reproduction/experiments/preprocess/build_full_maps.py, or "
                    "pass --allow-half-map if a half-map is genuinely intended."
                )
    # flags only the STRUCTURE route reads; on the map route they would be silently ignored.
    # `getattr` because a driver may declare a parser exposing only the flags it reads.
    if maps:
        for flag, attribute in (
            ("--structure-offset-from-maps", "structure_offset_from_maps"),
            ("--chains", "chains"),
            ("--no-superpose", "no_superpose"),
        ):
            if getattr(args, attribute, None):
                parser.error(
                    f"{flag} applies to the structure route and the map route was taken. A pair of "
                    "maps is already in a common frame and has no atoms to select or superpose, so "
                    "there is nothing for it to do here -- drop it, or pass --structure-a/-b."
                )
    if args.cs_pose_transpose is not None and str(args.meta).lower().endswith(".star"):
        parser.error("--cs-pose-transpose reads a cryoSPARC rotation vector; a .star already spells its Eulers out")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=Path, required=True, help="cryoSPARC .cs or RELION .star")
    parser.add_argument("--root", type=Path, required=True, help="directory the stored stack paths are relative to")
    parser.add_argument("--structure-a", type=Path, help="state A as an atomic model; superposed onto B")
    parser.add_argument("--structure-b", type=Path, help="state B as an atomic model")
    parser.add_argument(
        "--ligands",
        nargs="+",
        default=None,
        metavar="RESNAME",
        help="residue names to keep even where the two states differ, so a bound-ligand difference "
        "(two drugs, two peptides) reaches the templates. Name both states' ligands; empty keeps shared only",
    )
    parser.add_argument(
        "--map-a",
        type=Path,
        help="state A as a voxel map instead of a model, for ground truth that ships no atoms "
        "(CryoBench's conformational sets are volumes only). Not superposed; see `build_states`",
    )
    parser.add_argument("--map-b", type=Path, help="state B as a voxel map")
    parser.add_argument(
        "--allow-half-map",
        action="store_true",
        help="score against a half-map template anyway. A half-map's reconstruction noise is counted "
        "as signal by the oracle, inflating d^2 and widening the score variance; build a full template "
        "with reproduction/experiments/preprocess/build_full_maps.py instead",
    )
    parser.add_argument(
        "--known-pose",
        action="store_true",
        help="score each particle at the pose its metadata records instead of marginalizing over a "
        "grid. This is the d'^2_known UPPER BOUND, not the attainable d'^2_obs: marginalizing (the "
        "DEFAULT) is what a real experiment gets, and known-pose is run deliberately as the ceiling it "
        "is compared against. Needs a table carrying rlnAngle*. Also switches --calibrate to `poses`",
    )
    parser.add_argument(
        "--compare-rungs",
        action="store_true",
        help="with --known-pose, ALSO score the bank rung on the same stack, the same calibration, "
        "the same contrast and the same shift window, and record both. The chain bounds the bank by "
        "the oracle, so `known_pose_over_bank` is at least one and a value below it is a finding. "
        "Needed because comparing the two across separate runs cannot answer that: the routes "
        "calibrate differently by construction, so the contrasts differ and nothing is comparable",
    )
    parser.add_argument(
        "--calibrate",
        choices=("bank", "poses"),
        default=None,
        help="where the contrast and sigma^2(k) are fitted. DEFAULTS to `poses` under --known-pose and "
        "`bank` without it. `bank` marginalizes the pose over the bank and so biases the contrast LOW by "
        "however badly the bank matches (d^2 goes as a^2), and at a large box its (poses x shifts) model "
        "overflows memory -- while `poses` fits bank-free at each particle's recorded pose. `poses` is "
        "valid in the MARGINAL arm too: it calibrates at the recorded poses, then the SCORE still "
        "marginalizes over the bank -- the feasible, unbiased route for a large real pool with poses",
    )
    parser.add_argument(
        "--amplitude-shells",
        type=int,
        default=None,
        metavar="N",
        help="fit the template scale in N radial shells, a(k), instead of one scalar a. A scalar "
        "assumes the templates are wrong by the same factor at every frequency; an envelope or "
        "B-factor mismatch against whatever rendered the stack is a factor that varies with k, and "
        "least squares over the whole band absorbs it into an average that under-serves the "
        "orientations carrying more of their signal at high frequency. Keep N small -- 4 to 8 is an "
        "envelope, one shell per noise bin is a per-mode gain that starts fitting the difference "
        "signal itself. Needs --calibrate poses",
    )
    parser.add_argument(
        "--invert-contrast",
        action="store_true",
        help="negate the stored images, for a stack whose protein is bright where the forward model "
        "puts it dark. A `.cs` states this in blob/sign and a `.star` does not; scoring at the wrong "
        "sign fits a negative contrast and separates nothing, which the run refuses rather than reports",
    )
    parser.add_argument(
        "--state",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="keep only particles whose stack file name starts with one of these prefixes. A "
        "simulation benchmark names each stack for the structure that generated it, so this selects "
        "a pool of known composition out of a combined table. ONE prefix gives a single-state pool, "
        "where mean Lambda measures how well the templates match particles of a known state and "
        "nothing separates. TWO give a LABELED two-state pool, where the label is which file the "
        "particle came from -- the truth a benchmark gives away and a real collection never does -- "
        "so the pool's own separation becomes measurable and `shuffle_floor` has the labels it needs "
        "for a null. Order fixes which is state A, and it must match --structure-a/--map-a",
    )
    add_assembly_argument(parser)
    parser.add_argument(
        "--structure-offset-from-maps",
        nargs=2,
        type=Path,
        default=None,
        metavar="MAP",
        help="place the structure templates where these two reference maps put their density, rather "
        "than at the box center. For a simulated set whose images were generated from volumes "
        "carrying their own origin: CryoBench's IgG-1D volumes sit off center and its star "
        "recenters the particles onto that, so origin-centered templates would be displaced from the "
        "particles. Both maps are needed because the pair is centered on its two states AVERAGED, and "
        "state A's map alone under-corrects by half the inter-state shift. This is a rigid TRANSLATION "
        "read off the maps and nothing else -- the templates are still built from atoms, so the "
        "structure route remains an independent instrument",
    )
    parser.add_argument(
        "--no-superpose",
        action="store_true",
        help="take the two structures' frames as given instead of superposing B onto A. For a "
        "simulated set whose models are ALREADY in the frame its images were generated in -- "
        "CryoBench is one -- where superposing moves the templates off the particles. The structure "
        "route's analog of --map-frame none, and the two routes disagreeing is how you find out "
        "you needed it",
    )
    parser.add_argument(
        "--ignore-origin",
        action="store_true",
        help="accept a map whose header origin sits away from the box center. The guard reads a "
        "field two MRC conventions write differently -- a zero ORIGIN and nxstart = -N/2 both mean "
        "box-centered -- so it is a heuristic, and the deposited-map route offers the same hatch",
    )
    parser.add_argument("--out", type=Path, required=True, help="the result JSON")
    parser.add_argument("--log-ratio", type=Path, help="also write per-particle Lambda here, as .npy")
    parser.add_argument("--box-size", type=int, default=None, help="refuse a stack that is not this size")
    parser.add_argument("--pixel-size", type=float, default=None, help="override the file's, which can be wrong")
    parser.add_argument(
        "--dose",
        type=float,
        default=SHARED.dose,
        help="total exposure in e/A^2 the noise model is built for. 40 is the SIMULATED operating "
        "point; real data was collected at its own exposure and must pass it (the reproduce.py "
        "registry declares each real dataset's recorded dose), never inherit the simulated 40",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=None,
        help="inverse A the DEFOCUS GROUPING is built for. It does NOT stop the score -- that is "
        "--band-limit. Across a wide change here the fitted contrast and the SNR barely move, "
        "while the defocus-group count changes by more than an order of magnitude",
    )
    parser.add_argument(
        "--band-limit",
        type=float,
        default=None,
        help="highest frequency the score actually reaches, inverse A, as a hard cutoff on the "
        "transfer. Atomic templates are sharp and the render piles power just under Nyquist, so on "
        "a finely sampled stack the matched filter is dominated by modes the data cannot contain. "
        "Set it to where the refinement actually resolves",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="poses the CONTRAST and NOISE SPECTRUM are fitted against under --known-pose, which is why "
        "it needs it. The marginal arm ignores it -- it grows a fixed ladder to convergence (see "
        f"--cap-bank), not from --nodes. Defaults to {_DEFAULT_NODES}",
    )
    parser.add_argument(
        "--cap-bank",
        type=int,
        default=1_000_000,
        help="marginal arm only: grow the pose bank to convergence up to this cap instead of scoring one "
        "fixed --nodes. Each bank re-fits the calibration and re-scores; a low cap keeps a smoke test to "
        "one bank",
    )
    parser.add_argument(
        "--fixed-bank",
        type=int,
        default=None,
        metavar="K",
        help="marginal arm only: score ONE bank of this many nodes and skip adaptive convergence. For a "
        "per-K ladder run as separate jobs and extrapolated to K->inf offline (eq:bank), and for the "
        "100-particle convergence probe. The realized count is read back and recorded",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N particles. PER STATE when --state names more than one: a "
        "benchmark's star orders its stacks one after another, so a flat truncation of a two-state "
        "selection keeps only the first state and labels it as two",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        metavar="N",
        help="keep at most N particles, drawn UNIFORMLY AT RANDOM with --seed. Zero keeps all. NOT "
        "--limit, which takes the front of a table usually ordered by acquisition, so a truncated "
        "pool is one session's prefix. This is the flag for a cost cap on a pool that must stay "
        "representative: the --known-pose path renders a template per particle per state up front, "
        "so a large pool can ask substantial GPU memory inside calibrate_at_known_poses.",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=200,
        help="label permutations for the shuffle null on a labeled pool; more permutations resolve "
        "the null's tail more finely",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for the shuffle null")
    parser.add_argument(
        "--shift-window",
        type=int,
        default=0,
        metavar="PX",
        help="also marginalize translations, over a square window of +-PX whole pixels. 0 assumes "
        "the particles are centered, which is exact for a simulated stack extracted at its own "
        "coordinates and FALSE for anything picked and refined. Recentring by the table's own shifts "
        "leaves the refinement's residual error behind, and the E-step has no way to explain what is "
        "left except by shrinking the contrast and inflating sigma^2. So this BIASES rather than "
        "blurs. On a picked-and-refined collection the residual after recentring is a few px rms per "
        "axis, which +-3 to +-4 covers; the cost is one extra cross matmul per candidate, the pose "
        "bank never being rebuilt",
    )
    parser.add_argument(
        "--shift-step",
        type=float,
        default=1.0,
        metavar="PX",
        help="spacing of the shift candidates, in whole pixels. A transform reaches the pixel lattice "
        "and nothing between it, so this must divide the window into integers",
    )
    parser.add_argument(
        "--cs-pose-transpose",
        choices=("yes", "no"),
        default=None,
        help="which reading of a cryoSPARC `alignments3D/pose` to take, overriding "
        "`metadata.CS_POSE_TRANSPOSE`. `Rodrigues(-v) == Rodrigues(v).T`, so this is the single "
        "genuine degree of freedom in that conversion and the wrong one does NOT raise: it renders "
        "every template at a plausible but unrelated orientation, which reads as a small contrast "
        "and a bad specimen. Score a pool both ways and take the decisive margin, the way "
        "--check-shift-sign is used, rather than inheriting the constant onto a new dataset",
    )
    parser.add_argument("--no-recenter", action="store_true")
    parser.add_argument(
        "--null-images",
        action="store_true",
        help="phase-scramble each image (exact power kept, structure destroyed) before the known-pose "
        "score; mean(Lambda) is then the Einstein-from-noise class-independent midpoint",
    )
    parser.add_argument("--check-shift-sign", action="store_true", help="measure the recentring sign from the images")
    parser.add_argument("--refit-ctf", action="store_true", help="refit the defocus rather than trust the file")
    parser.add_argument(
        "--micrograph",
        type=Path,
        help="micrograph .mrc or .npy, required by --refit-ctf",
    )
    parser.add_argument(
        "--micrograph-pixel-size",
        type=float,
        default=None,
        help="A/px of the --micrograph itself, overriding the --meta file's "
        "location/micrograph_psize_A and required when the file has none. NOT the particle boxes', "
        "which extraction usually downsamples away from",
    )
    parser.add_argument("--n-iterations", type=int, default=60)
    args = parser.parse_args(argv)
    check_pool_arguments(parser, args)
    # --calibrate is resolved rather than defaulted, since neither choice is right in both cases:
    # without recorded poses `poses` has nothing to fit at, and with them `bank` biases d^2 low.
    # `poses` is valid in the MARGINAL arm too (it fits bank-free at the recorded poses, then the score
    # marginalizes) -- that is the only feasible route at a large box, where a per-bank fit overflows.
    if args.calibrate is None:
        args.calibrate = "poses" if args.known_pose else "bank"
    if args.known_pose and args.calibrate == "bank" and args.nodes is None:
        parser.error(
            "--known-pose with --calibrate bank needs --nodes stated explicitly. It does not use the bank "
            "to score -- each particle is rendered at its own recorded pose -- but the contrast and noise "
            "spectrum are fitted against the bank, and one too coarse to match anything fits a small "
            "contrast that scales the whole oracle down: 16 nodes gave a = 13.1 against 53.6 and put the "
            "marginal 3-5x ABOVE the ceiling bounding it. Pass the SAME --nodes as the marginal arm this "
            "run is the ceiling for, or pass --calibrate poses and be free of the bank entirely"
        )
    if args.amplitude_shells is not None:
        if not args.known_pose:
            parser.error(
                "--amplitude-shells needs --known-pose: a(k) scales the oracle rung's templates, and the "
                "marginal arm scores the bank directly, so a fitted envelope would not reach its score. "
                "Fit the envelope on the known-pose rung"
            )
        if args.calibrate == "bank":
            parser.error(
                "--amplitude-shells fits a(k) and must calibrate at the recorded poses, not the bank: a "
                "bank-fitted a(k) reads the bank's coarseness as an envelope, worst where the templates "
                "carry the most detail, and the two causes cannot be separated after the fact. Drop "
                "--calibrate bank -- poses is the default under --known-pose"
            )
        if args.amplitude_shells < 1:
            parser.error(f"--amplitude-shells must be at least one shell, got {args.amplitude_shells}")
    if args.nodes is None:
        args.nodes = _DEFAULT_NODES
    if args.refit_ctf and args.micrograph is None:
        parser.error("--refit-ctf needs --micrograph: there is nothing to refit a CTF from without one")
    if args.refit_ctf:
        # never the particle pixel size (the silent wrong answer) nor the micrograph header
        # (`read_micrograph` keeps none): the METADATA carries it, and a flag catches only "never stated"
        try:
            stated = stated_micrograph_pixel_size(args.meta)
        except ValueError as exc:
            parser.error(str(exc))
        if args.micrograph_pixel_size is None and stated is None:
            parser.error(
                "--refit-ctf needs --micrograph-pixel-size: this --meta states no "
                "location/micrograph_psize_A, and the table's own pixel size is the extracted boxes', "
                "which at a 256 -> 128 extraction is 2x wrong in every spatial frequency and ~4x "
                "wrong in the defocus that comes back"
            )
        if args.micrograph_pixel_size is None:
            args.micrograph_pixel_size = stated
        elif stated is not None and not np.isclose(args.micrograph_pixel_size, stated):
            # the flag wins (a deposition can be wrong), but a silent disagreement between two stated
            # numbers is said out loud
            print(
                f"--micrograph-pixel-size {args.micrograph_pixel_size:.4g} overrides the file's "
                f"{stated:.4g} A/px; the flag wins, and one of the two is wrong"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.known_pose:
        print(
            "WARNING: --known-pose scores each particle at its supplied pose, so this d^2 is the "
            "UPPER BOUND d^2_known. Pose marginalization (the default -- drop --known-pose) can only "
            "lower it; do not quote a known-pose d^2 as the discriminability an experiment attains."
        )
    meta, images, pixel_size, shift_report, labels = read_pool(args)
    # +1 on state A and -1 on state B, built ONCE. Three sites below orient by it -- the rung
    # comparison, the pool's own separation, and the shuffle null, which undoes it.
    is_a = None if labels is None else jnp.asarray(labels == 0)
    orientation = None if is_a is None else jnp.where(is_a, 1.0, -1.0)
    meta, ctf_report = refit_ctf(meta, args)

    box_size = int(images.shape[-1])
    plan = CollectionPlan(dose=args.dose, pixel_size=pixel_size)
    grid = pose_bank(args.nodes)
    models, states = build_states(args, plan, box_size, pixel_size)
    # per state and on demand: under `--known-pose` the state-B bank is never read and state A's only
    # reaches the printed matched-filter SNR, so rendering both up front would double the cost for nothing
    rendered: dict[int, jnp.ndarray] = {}

    def bank(state: int) -> jnp.ndarray:
        """The pose bank for one state, rendered at most once."""
        if state not in rendered:
            rendered[state] = models[state].bare_contrast(grid.angles)
        return rendered[state]

    # k_max is STATED rather than defaulted to Nyquist: it sets the defocus group count
    # quadratically, and a band the score never reaches buys groups nobody uses
    k_max = args.k_max if args.k_max else 0.5 / pixel_size
    # --band-limit is what actually STOPS the score at a frequency (--k-max sets only the defocus
    # grouping): the envelope multiplies the transfer, so zeroing it removes those modes from the score
    band = band_limit_envelope(box_size, pixel_size, args.band_limit) if args.band_limit else None
    transfer = transfer_from_meta(meta, box_size, pixel_size, k_max=k_max, envelope=band)

    # built ONCE, since a shift grid built per call is a fresh jit cache key per chunk; `None` (not an
    # empty grid) when the window is zero, so the scorer skips the transforms on a centered stack
    shifts = shift_grid(box_size, float(args.shift_window), step=args.shift_step) if args.shift_window else None

    # pose per particle is what `--known-pose` scores at and what `--calibrate poses` fits at; both need it
    if (args.known_pose or args.calibrate == "poses") and not meta.has_euler:
        raise ValueError(
            f"{'--known-pose' if args.known_pose else '--calibrate poses'} needs a pose per particle and "
            f"{meta.source} records none. A RELION star carries rlnAngleRot/Tilt/Psi; a cryoSPARC .cs "
            "carries `alignments3D/pose`, which only a 3D refinement writes; a 2D classification records "
            "an in-plane angle and no viewing direction, read as no pose at all. Drop the flag."
        )
    # rendered ONCE at each particle's recorded pose, for the known-pose rung and the known-pose
    # calibration. `--calibrate poses` fits the contrast, noise and shift width HERE, bank-free, so the
    # marginal score reuses one cheap unbiased calibration rather than re-fitting a (poses x shifts)
    # model per bank -- which overflows memory at a large box.
    templates_a: jnp.ndarray | None = None
    templates_b: jnp.ndarray | None = None
    fit = None
    if args.known_pose or args.calibrate == "poses":
        angles = jnp.asarray(euler_from_relion(meta.euler))
        templates_a, templates_b = models[0].bare_contrast(angles), models[1].bare_contrast(angles)
    if args.calibrate == "poses":
        # both states, never A alone: a single-state pool does not announce which it is; labels where present
        fit = calibrate_at_known_poses(
            images,
            jnp.stack([templates_a, templates_b]),
            transfer=transfer,
            state_of=None if labels is None else jnp.asarray(labels),
            shifts=shifts,
            n_amplitude_shells=args.amplitude_shells,
            n_iterations=args.n_iterations,
        )
        if shifts is not None:  # score under the fitted Gaussian width this calibration found
            shifts = _fitted_shift_grid(box_size, args, fit.shift_sigma)
            # a sub-grid width means the particles are centered (recentred residual below one pixel), so
            # marginalizing shifts only adds noise and forgoes the fast centered path -- score centered
            if float(fit.shift_sigma) < args.shift_step:
                print(
                    f"shift prior sigma {float(fit.shift_sigma):.3g} px is below the {args.shift_step} px "
                    "grid step: particles are centered, scoring without shift marginalization"
                )
                shifts = None

    # MARGINAL ARM: grow the pose bank until d^2 converges instead of scoring one fixed --nodes. Under
    # `--calibrate poses` the calibration above is fixed and only the SCORE is marginalized; under
    # `--calibrate bank` each rung re-fits (and re-finds the shift width), the blind-pool path, which is
    # memory-bound at a large box. Only the converged count is kept. `--known-pose` does not marginalize.
    convergence = None
    if not args.known_pose and args.fixed_bank:
        # score ONE bank and skip convergence: a per-K ladder run across jobs, extrapolated offline
        args.nodes = realized_nodes(args.fixed_bank)
        grid = pose_bank(args.fixed_bank)
    elif not args.known_pose:

        def _score_marginal(nodes: int) -> tuple[int, float]:
            grid_k = pose_bank(nodes)
            if fit is not None:
                noise_k, amp_k, shifts_k = fit.noise_spectrum, float(fit.contrast), shifts
            else:
                banks_k = jnp.stack([models[0].bare_contrast(grid_k.angles), models[1].bare_contrast(grid_k.angles)])
                fit_k = fit_noise_model(
                    images,
                    banks_k,
                    transfer=transfer,
                    pose_log_weights=grid_k.log_weights,
                    shifts=shifts,
                    n_iterations=args.n_iterations,
                )
                noise_k, amp_k = fit_k.noise_spectrum, float(fit_k.contrast)
                shifts_k = None if shifts is None else _fitted_shift_grid(box_size, args, fit_k.shift_sigma)
            marg = _marginal_scores(models, grid_k, images, noise_k, amp_k, shifts_k, transfer, box_size)
            # the deflection d^2 = gap^2/within on a labeled pool; single pool has no gap, so track |mean|
            if orientation is not None and len(args.state) > 1:
                d2 = float(deflection(marg[is_a], marg[~is_a]))
            else:
                d2 = float(2.0 * jnp.abs(jnp.mean(marg)))
            return realized_nodes(nodes), d2

        convergence = converge_marginal(_score_marginal, cap=args.cap_bank)
        args.nodes = max(convergence.curve)
        grid = pose_bank(args.nodes)

    # `--calibrate bank` fits against the converged bank (marginal) or the stated --nodes (known-pose),
    # on the same shift window the scorer uses; `--calibrate poses` already fitted above, bank-free.
    if args.calibrate == "bank":
        fit = fit_noise_model(
            images,
            jnp.stack([bank(0), bank(1)]),
            transfer=transfer,
            pose_log_weights=grid.log_weights,
            shifts=shifts,
            n_iterations=args.n_iterations,
        )
        if shifts is not None:  # score under the fitted Gaussian width this calibration found
            shifts = _fitted_shift_grid(box_size, args, fit.shift_sigma)
    # a NEGATIVE contrast means every template anti-correlates with its data (a contrast convention
    # opposite to the templates'), which does not announce itself: the fit converges and pools do not separate
    if float(fit.contrast) < 0.0:
        raise ValueError(
            f"the calibration fitted a NEGATIVE contrast, a = {float(fit.contrast):.4g}. Every "
            "template anti-correlates with its own particles, which means this stack stores the "
            "opposite contrast convention from the templates -- protein bright where the forward "
            "model puts it dark, as simulated benchmarks often do. Pass --invert-contrast. A "
            "`.cs` pool states the convention in `blob/sign` and a `.star` does not, so this is "
            "measured from the fit rather than read from the file."
        )
    # a(k) enters by SCALING THE TEMPLATES, not as an `amplitude=` argument: `a(k) * T` is the identical
    # model, and doing it here keeps the score and its closed form on one path that agrees on the contrast
    contrast, amplitude_shells, envelope = float(fit.contrast), None, None
    if fit.amplitude_spectrum is not None:
        envelope = fit.amplitude_spectrum[None]
        templates_a = envelope * templates_a
        templates_b = envelope * templates_b
        contrast = 1.0
        # read back per shell for the record: a(k) is constant within a shell, so a mean recovers the
        # fitted value exactly. The last bin is the grid corners (outside every shell), reported separately.
        index = np.asarray(shell_indices(box_size, args.amplitude_shells)[0]).reshape(-1)
        centers = [float(k) for k in shell_indices(box_size, args.amplitude_shells)[1]]
        flat = np.asarray(fit.amplitude_spectrum).reshape(-1)
        amplitude_shells = {
            "n_shells": args.amplitude_shells,
            "k_cyc_px": [*centers, None],
            "k_inverse_angstrom": [*(k / pixel_size for k in centers), None],
            "a": [
                float(flat[index == j].mean()) if int((index == j).sum()) else None
                for j in range(args.amplitude_shells + 1)
            ],
            "n_modes": [int((index == j).sum()) for j in range(args.amplitude_shells + 1)],
        }

    def calibrated_bank(state: int) -> jnp.ndarray:
        """The pose bank carrying the same contrast the oracle's templates do.

        a(k) is folded into `templates_a`/`templates_b` above and `contrast` reset to 1.0, so a rung
        that reads `bank()` directly and passes that same 1.0 is scored with NO calibration at all --
        which `--compare-rungs` must not do, since it holds everything but the pose description fixed.
        """
        return bank(state) if envelope is None else envelope * bank(state)

    baseline, oracle_closed_form = 1.0, None
    if args.null_images:
        # EINSTEIN-FROM-NOISE null: keep each image's power spectrum but randomize its phases, so
        # mean(Lambda) becomes the class-independent MIDPOINT (m_A+m_B)/2 -- a no-signal control
        img = np.asarray(images)
        rng = np.random.default_rng(args.seed)
        rphase = np.fft.fft2(rng.standard_normal(img.shape))
        rphase /= np.abs(rphase) + 1e-30
        images = jnp.asarray(np.abs(img) * rphase)
    if args.known_pose:
        # the ORACLE rung, which needs a pose per particle and so is unavailable on a real pool; a
        # simulation benchmark ships it, and it is the ceiling every other rung is quoted against
        marginal = known_pose_log_ratio(
            images,
            templates_a,
            templates_b,
            fit.noise_spectrum,
            amplitude=contrast,
            transfer=transfer,
            # knowing the pose does not center the particle; without this the oracle rung reads a
            # near-zero coherence on a picked-and-refined pool
            shifts=shifts,
        )
        # the CLOSED FORM, from the same templates and no stack: with no pose search `d_i^2` is fully
        # determined, so the pool's coherence has a predicted baseline (below one, the poses varying)
        d2 = known_pose_d2(templates_a, templates_b, fit.noise_spectrum, amplitude=contrast, transfer=transfer)
        baseline = predicted_coherence(d2)
        # `d2_mean` and NOT a signed `predicted_mean`: `E[Lambda|B] = -E[d^2]/2`, so a positive
        # predicted mean would disagree in sign with `coherence.mean` on every state-B pool
        oracle_closed_form = {
            "d2_mean": float(jnp.mean(d2)),
            "d2_spread": float(jnp.var(d2)),
            # the pool ceiling d'^2_known = E[D]^2/(E[D]+Var[D]/4), the deflection of the known-pose
            # pool (derivations §7); below E[D] when poses vary. This is the bound d'^2_obs clears.
            "d2_known_pool": float(jnp.mean(d2) ** 2 / (jnp.mean(d2) + jnp.var(d2) / 4.0)),
            "predicted_coherence": baseline,
        }
        if args.compare_rungs:
            # the SAME calibration, contrast, shift window and stack, differing only in which pose
            # description the score integrates -- separate runs cannot compare, calibrating differently
            bank_marginal = pose_log_ratio(
                images,
                calibrated_bank(0),
                calibrated_bank(1),
                fit.noise_spectrum,
                amplitude=contrast,
                pose_log_weights=grid.log_weights,
                shifts=shifts,
                transfer=transfer,
            ).marginal
            # ORIENTED by the pool, not per rung (they score the same particles): the mean is signed and
            # `E[Lambda|B] = -d^2/2`, so an unoriented `chain_holds` on a state-B pool compares two negatives
            if orientation is not None and len(args.state) > 1:
                rung_sign = orientation
            else:
                rung_sign = jnp.asarray(1.0 if float(jnp.mean(marginal)) >= 0.0 else -1.0)
            at_known_pose = coherence(marginal * rung_sign)
            at_bank = coherence(bank_marginal * rung_sign)
            # the chain is about `d^2` (`E[Lambda] = +d^2/2` on a pure pool), so the MEANS compare;
            # coherence is not monotone in information, so the chain ratio is built from the means
            oracle_closed_form["rung_comparison"] = {
                "known_pose_d2": 2.0 * at_known_pose.mean,
                "bank_d2": 2.0 * at_bank.mean,
                "known_pose_over_bank": float(at_known_pose.mean / at_bank.mean),
                "chain_holds": bool(at_known_pose.mean >= at_bank.mean),
                # kept beside them because a large gap here with the chain intact is a CALIBRATION
                # finding, and it is the only place the two are measured under one contrast
                "known_pose_coherence": at_known_pose.value,
                "bank_coherence": at_bank.value,
                "n_nodes": int(args.nodes),
            }
    else:
        # streamed so the converged bank (which can be very large on a real pair) never materializes whole
        marginal = _marginal_scores(
            models, grid, images, fit.noise_spectrum, float(fit.contrast), shifts, transfer, box_size
        )

    # a labeled two-state pool's states cancel in the raw mean (`E[Lambda|A] = +d^2/2 = -E[Lambda|B]`),
    # so orienting by `s = +1` on A, `-1` on B gives `E[Lambda * s] = d^2/2`, a one-state pool's scale
    separation: dict[str, Any] | None = None
    if is_a is not None and len(args.state) > 1:
        oriented = marginal * orientation
        separation = {
            "n_a": int(jnp.sum(is_a)),
            "n_b": int(jnp.sum(~is_a)),
            "mean_lambda_given_a": float(jnp.mean(marginal[is_a])),
            "mean_lambda_given_b": float(jnp.mean(marginal[~is_a])),
            # d^2 = gap^2/within from both halves; the mean gap is a diagnostic, = d^2 only at a known pose
            "d2_pool_labeled": float(deflection(marginal[is_a], marginal[~is_a])),
            "mean_gap_labeled": float(jnp.mean(oriented) * 2.0),
        }
        print(
            f"labeled pool  {separation['n_a']} of state A, {separation['n_b']} of state B\n"
            f"mean Lambda   {separation['mean_lambda_given_a']:+.4g} given A, "
            f"{separation['mean_lambda_given_b']:+.4g} given B   "
            f"(opposite signs is the instrument working)"
        )
        marginal = oriented

    read = coherence(marginal, baseline=baseline)
    if oracle_closed_form is not None:
        # oriented by the pool's own SIGN, since an unlabeled state-B pool has `E[Lambda] = -d^2/2` and
        # a fractional error on the raw array would read -2 on a working instrument
        oriented_for_delta = marginal * (1.0 if read.mean >= 0.0 else -1.0)
        scale, spread = fractional_error(oriented_for_delta, d2)
        # the split `decoherence` cannot make: the MEAN is a scale error (duplicating
        # `implied_amplitude_scale`), the SD the open quantity, a fraction comparable across pools
        oracle_closed_form["fractional_error_mean"] = scale
        oracle_closed_form["fractional_error_sd"] = spread
    # a coarse bank: the matched-filter SNR is a per-template diagnostic, so it needs no fine grid, and
    # rendering the converged marginal bank whole would overflow memory at a large box
    snr_bank = (
        bank(0)
        if (args.known_pose or args.nodes <= _DEFAULT_NODES)
        else models[0].bare_contrast(pose_bank(_DEFAULT_NODES).angles)
    )
    snr = matched_filter_snr(snr_bank, fit.noise_spectrum, transfer=transfer, contrast=float(fit.contrast))

    print(f"particles   {len(meta)}   box {box_size}   pixel {pixel_size:.4g} A")
    print(
        f"calibration a = {float(fit.contrast):.4g} at the {args.calibrate}   SNR {snr:.3g}   "
        f"defocus groups {transfer.values.shape[0]}"
    )
    # the SHAPE on screen, since which WAY a(k) goes is the result: a FALL means the templates carry
    # more high-frequency power than the data (a missing envelope). The word is chosen from the ratio.
    if amplitude_shells is not None:
        usable = [a for a in amplitude_shells["a"] if a is not None]
        shape = "  ".join("--" if a is None else f"{a:.3g}" for a in amplitude_shells["a"])
        print(f"            a(k) over {args.amplitude_shells} shells + corner:  {shape}")
        if usable[0] > 0.0:
            ratio = usable[-1] / usable[0]
            way = "FALLS" if ratio < 1.0 else "RISES"
            reading = (
                "the templates carry MORE high-frequency power than the data, so the forward model is "
                "missing an envelope"
                if ratio < 1.0
                else "the templates carry LESS high-frequency power than the data"
            )
            print(f"            a(k) {way} {ratio:.3g}x from the first shell to the corner -- {reading}")
        else:
            print("            a(k) is non-positive in the first shell, so no shape can be read off it")
        # a NEGATIVE shell (templates anti-correlating there) no envelope error can produce, so the
        # shell was fitted on too little -- which the scalar guard misses, its power-weighted mean staying positive
        if min(usable) <= 0.0:
            print(
                f"            ^ {sum(a <= 0.0 for a in usable)} of {len(usable)} shells came out NEGATIVE. "
                "An envelope rescales the templates, it cannot invert them, so this is a shell count "
                "this pool does not support. Do not read a spectrum off it -- use fewer shells, and "
                "note that d^2 still comes out positive either way, so nothing downstream will complain"
            )
    # said on screen, not only in the help, because the bank route reads d^2 low and its error does
    # not announce itself in any other line of this output
    if args.known_pose and args.calibrate == "bank":
        print(
            "            ^ fitted against the BANK by explicit --calibrate bank, which reads LOW by "
            "however badly it matches: 2.0x in d^2 at 8192 nodes and 4.2x at 1024 on CryoBench, and "
            f"--nodes here is {args.nodes}. Do NOT quote an absolute d^2 from this run; drop the flag "
            "to fit at the recorded poses, which is the default and is also cheaper"
        )
    if oracle_closed_form is None:
        print(f"coherence   {read.value:+.3f}   (a marginal pool has no closed-form baseline; 1.000 is the ceiling)")
    else:
        # the pool's own d^2 against the closed form's says what the contrast was wrong BY, and since
        # the baseline is not scale-free that error propagates in; recomputed at the implied scale it brackets
        closed = oracle_closed_form["d2_mean"]
        implied_scale = (2.0 * abs(read.mean) / closed) ** 2
        implied_baseline = predicted_coherence(d2 * implied_scale)
        oracle_closed_form |= {
            "d2_pool": 2.0 * abs(read.mean),
            "implied_amplitude_scale": float(np.sqrt(implied_scale)),
            "predicted_coherence_at_implied_amplitude": implied_baseline,
        }
        print(
            f"coherence   {read.value:+.3f}   against a predicted {baseline:.3f} from the templates "
            f"(decoherence {read.decoherence:+.3f})"
        )
        print(
            f"d^2         closed form {closed:.4g}   pool {2 * abs(read.mean):.4g}   "
            f"({200 * abs(read.mean) / closed - 100:+.2f}%)"
        )
        print(
            f"            that gap puts the true contrast {float(np.sqrt(implied_scale)):.3f}x the fitted "
            f"one, at which the predicted coherence is {implied_baseline:.3f} rather than {baseline:.3f} "
            f"-- the baseline is NOT scale-free, so read the decoherence between the two"
        )
        # what the decoherence IS: the mean is the scale error on the line above, the sd the part no
        # rescaling reaches, a fraction comparable across pools
        print(
            f"fractional  error on d_i^2: mean {oracle_closed_form['fractional_error_mean']:+.3f}, "
            f"sd {oracle_closed_form['fractional_error_sd']:.3f} with the score's own noise subtracted "
            f"-- the sd is the decoherence, and only the mean touches a quoted d^2"
        )
    print(f"mean Lambda {read.mean:+.4g}")

    # written BEFORE the scale-error refusal deliberately: a refused pool's Lambda is what a downstream
    # analysis needs to recover the scale error, so it must survive the refusal (the run still fails)
    if args.log_ratio:
        args.log_ratio.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.log_ratio, np.asarray(marginal))
        # per-particle oracle d_i^2, aligned with Lambda, for pool-free d^2-stratification: the d^4
        # coefficient of E[Lambda^2 | d^2] gives (2p-1)^2 without labels ((2z-1)^2 = 1).
        if args.known_pose:
            np.save(args.log_ratio.with_name(args.log_ratio.stem + "_d2.npy"), np.asarray(d2))

    # ONLY at the oracle rung: `2E/Var <= 1` is derived for the single-component Gaussian LLR, not the
    # pose-marginal logsumexp, so a marginal reading above one is said loudly but does not fail the run
    if read.indicates_scale_error():
        message = (
            f"the pool reads a coherence of {read.value:+.4g}, above the 1.0 that a correctly scored "
            f"pool cannot exceed -- spread in d^2 and pool impurity BOTH only push it DOWN. That points "
            f"at a scale error: Lambda too small by AT MOST about {read.scale_error:.3g} (an upper "
            f"bound, assuming the two templates carry equal whitened power; correct it with "
            f"Coherence.use_power_asymmetry when they do not). The usual cause is a calibration "
            f"fitted against a bank too coarse to match the data, so the contrast (here a = "
            f"{float(fit.contrast):.4g}) comes out low; --nodes {args.nodes} may be too coarse for "
            f"this box. Pass more --nodes, or --k-max to band-limit the score."
        )
        if args.known_pose:
            raise ValueError(message)
        print(
            f"\n!! {message}\n   Refusing only at the oracle rung, where the identity is derived; "
            "this is the marginal rung, so the run continues and the d^2 above is NOT quotable."
        )

    if separation is not None:
        # permuting the labels leaves the pose/defocus spreads and noise intact and removes only the
        # association with the truth (what is tested); `marginal` is ORIENTED here, so it permutes the signs
        floor = shuffle_floor(
            marginal * orientation,  # undo the orientation
            is_a,
            key=jax.random.key(args.seed),
            n_repeats=args.n_shuffles,
        )
        separation["null"] = {
            # the statistic the null permutes, stated by `shuffle_floor` itself rather than rebuilt here
            "observed": float(floor.observed) if floor.observed is not None else None,
            "mean": float(floor.mean),
            "sd": float(floor.sd),
            "n_repeats": args.n_shuffles,
            # sigmas above the null (below ~2 it is indistinguishable from label noise); from
            # `floor.z(observed)`, not `d2_pool_labeled`, which is a differently weighted statistic
            "sigmas": float(floor.z(floor.observed)) if floor.sd > 0 and floor.observed is not None else None,
        }
        print(
            f"shuffle null  {floor.mean:+.4g} +/- {floor.sd:.4g} over {args.n_shuffles} permutations"
            f"   -> the measurement sits {separation['null']['sigmas']:.1f} sigma above it"
            if separation["null"]["sigmas"] is not None
            else f"shuffle null  {floor.mean:+.4g} with zero spread, which is a degenerate null"
        )

    write_result(
        args.out,
        {
            "n_particles": len(meta),
            "box_size": box_size,
            "pixel_size": pixel_size,
            "k_max": k_max,
            "n_defocus_groups": int(transfer.values.shape[0]),
            "amplitude": float(fit.contrast),
            # the SHAPE is the result, recorded per shell not summarized; `None` where a scalar was
            # fitted, so the two runs stay distinguishable in the record
            "amplitude_spectrum": amplitude_shells,
            # WHERE the contrast was fitted: `bank` carries a --nodes bias `poses` does not, so a
            # consumer comparing them across runs needs to know which it has
            "calibrated_at": args.calibrate,
            # what the calibration made of the pool's composition; on `poses` with an unlabeled pool it
            # is FITTED, and a pure pool reading a split occupancy means the templates cannot separate it
            "calibration_state_weights": np.asarray(fit.state_weights).tolist(),
            "snr": snr,
            "coherence": {
                "value": read.value,
                "mean": read.mean,
                "variance": read.variance,
                # what a correctly scored pool of THESE particles must read. One when marginalizing,
                # where no closed form is available, and the templates' own prediction at the oracle
                "baseline": read.baseline,
                "decoherence": read.decoherence,
            },
            "oracle_closed_form": oracle_closed_form,
            # `None` on a single-state pool, where there is nothing to separate and so no null
            # either. A consumer should refuse to quote a separation without this beside it
            "separation": separation,
            "shift": shift_report,
            # the EM-fitted Gaussian shift-prior width (px) the scorer marginalized under; None centered
            "shift_prior_sigma": float(fit.shift_sigma) if args.shift_window else None,
            "ctf_refit": ctf_report,
            # WHICH two states this pool was scored against and by which route; a log-ratio is
            # meaningless without them, and `arguments` alone does not say if the pair was superposed
            "states": states,
            "source": meta.source,
            "log_ratio_path": str(args.log_ratio) if args.log_ratio else None,
            # the pose-bank convergence the marginal arm settled on; None on --known-pose, which scores
            # at recorded poses and does not marginalize over a bank
            "convergence": None if convergence is None else convergence.as_record(),
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
