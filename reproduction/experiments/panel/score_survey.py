"""Rank a panel of two-state targets by the particle count needed to separate them.
Physics in, particle count out. Each pair is built from its two entries, scored at ONE stated
collection plan, and the Cramer-Rao bound inverted to the count that pins the population fraction.
    .venv/bin/python reproduction/experiments/panel/score_survey.py --data-dir data/pdb --fetch
Two counts per row: `n_floor` from the model-only rungs (pose assumed known) is always available;
`n_marginal` costs a simulated pool per row and is the number to quote, turned on by `--marginalize`
(hours), else every quotable count is marked a floor. `--fraction` sets the operating point: `I_p`
caps at `1/[p(1-p)]`, so the count bottoms out at `p(1-p)/sigma^2` and tightening `--sigma` cannot
help. A rejected candidate is reported, not dropped, else the table looks like everything tried.
"""

import argparse
import time
from pathlib import Path

import jax

from cryodisc import CollectionPlan
from cryodisc.grids.poses import pose_bank
from cryodisc.statistics.convergence import Extrapolation, RefusedExtrapolation
from reproduction.chain import run_chain
from reproduction.operating_point import SHARED
from reproduction.record import write_result
from reproduction.targets import (
    _MAX_PLAUSIBLE_RMSD,
    Candidate,
    TargetRow,
    attach_marginal,
    format_table,
    load_panel,
    rank_targets,
)
from reproduction.validation.floor import twin_floor

DRIVER = "reproduction/experiments/panel/score_survey.py"


def measure_marginal(
    row: TargetRow, candidate: Candidate, plan: CollectionPlan, args: argparse.Namespace
) -> tuple[TargetRow, dict]:
    """Run the chain for one row and attach its marginal rung.

    The expensive arm, and the only one whose count is quotable. The pair is rebuilt through
    `Candidate.align`, which is the one call `score_target` makes too, rather than carried on the
    row -- so the two arms cannot end up describing different assemblies of the same candidate,
    and a field added to `Candidate` reaches both or neither.

    Args:
        row: The row from the cheap arm, whose `box_size` this reuses so both arms share a grid.
        candidate: The candidate it came from.
        plan: The collection.
        args: Parsed arguments, for the pool size, the grids and the error target.

    Returns:
        `(row, null)`. The row has `n_marginal` filled, or is unchanged if the chain could not run
        or the rung it produced was refused -- one failed row must not take a panel down. `attach_marginal`
        is INSIDE the guard because its oracle check is the likeliest failure and the one a panel most
        needs to survive. `null` carries the twin floor when `--twin-null` is on, and travels beside
        the row rather than on it so a row is never quoted without the null in the same file.
    """
    null: dict = {"measured": False}
    # whatever has been measured so far, so a failure in a LATER stage reports the rung already
    # computed -- returning the bare input row threw away hours of chain compute
    measured = row
    try:
        # min_shared_fraction MUST match the cheap arm: the module default re-refuses here what
        # `rank_targets` was told to admit, and the blanket handler logs that as a failed rung
        pair = candidate.align(
            hydrogens=getattr(args, "ablate", None) == "hydrogens",
            min_shared_fraction=args.min_shared_fraction,
        )
        # `Extrapolation.check` needs a wide span for the exponent to be identifiable; under `--poses
        # bank` the ladder measures divisibility, not convergence, so `haar` is the default. Sorted ascending.
        grids = [pose_bank(n) for n in sorted(args.extrapolate_nodes)] if args.extrapolate else None
        # `run_chain` produces the extrapolation and does not grade it, so the ladder is in hand
        # whichever way the check goes -- which lets the refusal below fall back to the raw marginal
        result = run_chain(
            pair,
            plan,
            key=jax.random.key(args.seed),
            box_size=row.box_size,
            rmsf=args.rmsf,
            n_particles=args.n_particles,
            error_target=args.sigma,
            # the SAME fraction the cheap arm quoted its floor at, since `attach_marginal` checks the two
            # rungs against each other and a marginal at one occupancy against a floor at another compares p's
            fraction=args.fraction,
            marginal_grids=grids,
            poses=args.poses,
            # non-zero puts a translation latent in the marginal rung, so the survey marginalizes
            # the same nuisance parameters the real-data arms do; zero draws every particle centered
            shift_sigma=args.shift_sigma,
            form_factor="z" if args.ablate == "form-factor" else "peng",
        )
        limit = result.marginal_limit
        try:
            if limit is not None:
                # `known_pose`, not a bank-dependent rung: a ceiling on the coarse base bank
                # understates the true one and would not bound a rung extrapolated to K -> infinity
                limit.check(ceiling=result.known_pose.d2)
            rung = float(limit.limit if limit is not None else result.marginal.d2)
            rung_err = float(limit.limit_err if limit is not None else result.marginal.d2_err)
            extrapolated = limit is not None
        except RefusedExtrapolation as refusal:
            # the K^(-2/3) fit was refused, so there is no converged number: quote the raw marginal at
            # the base bank as the LOWER BOUND it is (bounded above by `known_pose`), marked not extrapolated
            rung = float(result.marginal.d2)
            rung_err = float(result.marginal.d2_err)
            extrapolated = False
            print(
                f"  extrapolation refused ({str(refusal)[:60]}...); "
                f"quoting the raw marginal {rung:.4g} as a lower bound",
                flush=True,
            )
        # the rung's error bar travels WITH it: the oracle check is an inequality between two
        # numbers of which one is a pool estimate, and on bank the two are the same number
        measured = attach_marginal(
            row,
            rung,
            d2_marginal_err=rung_err,
            extrapolated=extrapolated,
            error_target=args.sigma,
            fraction=args.fraction,
            # the Fisher information I(p) that sets `n_marginal`, from the PRIMARY grid, so consistent
            # with `rung` only when NOT extrapolated; None there falls back to the Gaussian count
            information=float(result.marginal.information) if not extrapolated else None,
        )
        if args.twin_null:
            # a DIFFERENT key, since a twin sharing the measurement's noise realization is correlated
            # with what it bounds; `twin_floor` reuses the rest of the operating point
            twin = twin_floor(result, pair, plan, key=jax.random.key(args.seed + 1))
            null = {
                "measured": True,
                # the oracle rung, which bounds every other: if the twin reads anything HERE the
                # floor is in the forward model itself and not in the pose search or the fit
                "known_pose_d2": float(twin.known_pose.d2),
                "marginal_d2": float(twin.marginal.d2),
                "marginal_d2_err": float(twin.marginal.d2_err),
                "measurement_marginal_d2": float(result.marginal.d2),
                # the d^2 rungs above are zero BY CONSTRUCTION (state B is the same atom table as A), so
                # they are a correctness check; the fit is the informative part, measuring the estimator's bias
                "fraction_bias": float(twin.fraction_bias),
                "calibration_expected_error": float(twin.calibration.expected_error),
                "d2_rungs_are_zero_by_construction": True,
            }
            worst = max(abs(twin.known_pose.d2), abs(twin.marginal.d2))
            print(
                f"  twin null: d^2 rungs {worst:.3g} (zero expected), fraction bias "
                f"{twin.fraction_bias:+.4g} against the measurement's {result.fraction_bias:+.4g}"
            )
        # record the fit that produced the rung so a later reader needs no cluster time to see its
        # ladder, exponent and chi2
        if extrapolated and limit is not None:
            null = null | {"extrapolation": {"method": "K^-p fit"} | _fit_dict(limit)}
        return measured, null
    except Exception as error:  # noqa: BLE001 - a panel row must fail alone
        print(f"  marginal rung failed for {row.name}: {error}")
        return measured, null | {"error": str(error)} | _ladder_of(error)


def _ladder_of(error: BaseException) -> dict:
    """The ladder a refused extrapolation was refused for, as a dict, or nothing.

    `RefusedExtrapolation` carries its fit so a refusal does not throw away the pool scores behind it
    -- half an hour of CPU per panel row, against a re-fit that costs milliseconds. This is what puts
    those points in the result file, so the next question about a refused row needs no cluster time.

    Args:
        error: Whatever came out of the guarded block. Anything that is not a `RefusedExtrapolation`
            -- an OOM, a missing file, the oracle check in `attach_marginal` -- carries no fit and
            gets an empty dict, its message being recorded separately.

    Returns:
        `{"ladder": {...}}`, or `{}` when there is no ladder to record.
    """
    fit = getattr(error, "extrapolation", None)
    if fit is None:
        return {}
    try:
        return {"ladder": _fit_dict(fit)}
    except AttributeError:
        # best effort inside the per-row handler: a stale or half-written install could raise from
        # `_fit_dict` and take the whole panel down, and losing one row's ladder is the smaller loss
        return {"ladder": {"unavailable": "this Extrapolation does not carry the fields recorded here"}}


def _fit_dict(fit: Extrapolation) -> dict:
    """One `Extrapolation` as a record, written once for both places a row can carry one.

    A refused fit is recorded under `ladder` and a successful one under `extrapolation`, and the two
    were spelled out separately over seven shared keys -- so the same fit could be described two ways
    depending on which branch a row took.

    Args:
        fit: The extrapolation.

    Returns:
        Its fields, JSON-ready.
    """
    return {
        "limit": float(fit.limit),
        "limit_err": float(fit.limit_err),
        "slope": float(fit.slope),
        "exponent": float(fit.exponent),
        "chi2_per_dof": float(fit.chi2_per_dof),
        "weighted": bool(fit.weighted),
        "span": float(fit.span),
        "counts": list(fit.counts),
        "values": list(fit.values),
        "errors": list(fit.errors),
        # the exponent each consecutive pair implies: the assumed 2/3 is not identifiable over a short
        # span, so a successful fit is only as trustworthy as these are flat
        "local_exponents": [float(value) for value in fit.local_exponents],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path(__file__).parent / "panel.json")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--fetch", action="store_true", help="download missing depositions from RCSB")
    parser.add_argument("--out", type=Path, default=Path("results/target_ranking.json"))
    parser.add_argument("--pixel-size", type=float, default=SHARED.pixel_size)
    parser.add_argument("--dose", type=float, default=SHARED.dose)
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument(
        "--min-shared-fraction",
        type=float,
        default=None,
        help="overlap below which a correspondence is refused, default 0.75. Lower it (0.0 admits "
        "anything with a non-empty intersection) to SCORE a partial correspondence instead of "
        "discarding it; the row records shared_fraction, so how much molecule a d^2 needs becomes a "
        "filter on the scored population. A d^2 over 60%% of a molecule is not comparable to one "
        "over 99%%, so those rows must be filtered on that field and not trusted because they exist",
    )
    parser.add_argument(
        "--no-rmsd-guard",
        action="store_true",
        help="do not refuse a pair on Ca RMSD. The guard only fires where residue-type identity "
        "cannot be computed. `identity` is the test it stands in for and is recorded per row",
    )
    parser.add_argument("--rmsf", type=float, default=SHARED.rmsf, help="per-axis intra-state RMS fluctuation in A")
    parser.add_argument("--sigma", type=float, default=0.02, help="target error on the population fraction")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="population fraction of state A, quoted and simulated. Sets the counting floor "
        "p(1-p)/sigma^2 the N column saturates on; see `targets.format_table`",
    )
    parser.add_argument("--budget", type=int, default=100_000, help="particles that count as achievable")
    parser.add_argument("--marginalize", action="store_true", help="measure the marginal rung per row (hours)")
    parser.add_argument(
        "--poses",
        choices=("bank", "haar"),
        default="haar",
        help=(
            "where the simulated particles' true poses come from. `haar` draws from the true "
            "measure and is the DEFAULT; `bank` draws them from the nodes of `run_chain`'s own grid, "
            "which isolates the fitting loss and is a diagnostic rather than a measurement. "
            "The banks NEST: a geodesic sphere contains every sphere at a triangulation "
            "number dividing its own, and psi is a uniform cycle, so a pool drawn ON one bank sits "
            "at zero miss from some rungs of a ladder and not others -- the miss then rises and "
            "falls across the ladder instead of falling, and a fit in K over it measures which node "
            "counts divide the pool's grid rather than how the rung converges. "
            "`chain._extrapolate_marginal` refuses such a ladder rather than fitting it, so with "
            "`--extrapolate` on, `bank` does not merely bias the row, it stops it."
        ),
    )
    parser.add_argument(
        "--extrapolate",
        action="store_true",
        help="with --marginalize, run three grids and fit K^-p, p = CROSSOVER_EXPONENT",
    )
    parser.add_argument(
        "--extrapolate-nodes",
        type=int,
        nargs="+",
        default=[504, 1840, 7056, 33384],
        metavar="N",
        help="APPROXIMATE node counts of the banks --extrapolate fits over, ascending. Approximate "
        "because `poses.pose_bank` quantizes: it returns the bank nearest the request, so read the "
        "sizes back rather than assuming these -- though this default is chosen so all four are hit "
        "EXACTLY. Two constraints a hand-picked ladder easily misses. FIRST, the pool must be drawn "
        "from Haar -- `--poses haar`, the default -- because the banks NEST: psi is a uniform cycle "
        "and a geodesic sphere contains every sphere at a dividing triangulation number, so a pool "
        "sitting ON one bank sits at zero miss from some rungs and not others, and the fit then "
        "measures divisibility rather than refinement. `chain._extrapolate_marginal` refuses such a "
        "ladder rather than fitting it. SECOND, no point may be coarse: a coarsest rung that does not "
        "fit leaves `Extrapolation.check` refusing the ladder, with pairwise exponents of nan, "
        "meaning a measured point sat at or beyond the fitted limit",
    )
    parser.add_argument(
        "--twin-null",
        action="store_true",
        help="re-run the chain on IDENTICAL states, where there is provably nothing to find, and "
        "record what comes back. Doubles the marginal arm's cost. Read it for what it is: the d^2 "
        "rungs are zero BY CONSTRUCTION because the twin's two states are one object, so they are a "
        "correctness check -- anything nonzero is a bug -- and NOT the estimator floor that makes a "
        "rung quotable. The FIT side is the real measurement: asked to split a population between "
        "two indistinguishable states, where the fit puts the fraction is its own bias. A different "
        "PRNG key, since a null sharing the measurement's noise realization is correlated with what "
        "it bounds",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="score only these panel rows, named exactly as `panel.json` does. The marginal arm is "
        "hours per row and sequential, so this is what lets one row per job run them in parallel and "
        "what makes a panel RESTARTABLE: a row that dies takes only itself, and rerunning it does not "
        "rerun the fifteen that succeeded. A name the panel does not contain is an error listing the "
        "ones it does, since a typo would otherwise silently score nothing",
    )
    parser.add_argument(
        "--ablate",
        default=None,
        choices=("hydrogens", "form-factor"),
        help="score the panel with one simplification applied, for E7. The measurement is a FLIP "
        "RATE, not a magnitude: run the panel twice, once with this and once without, and feed both "
        "to a decision-flip comparison, which keys the two scorings by row name and counts "
        "reversed orderings stratified by how much the ordering mattered. `hydrogens` KEEPS modeled "
        "hydrogens, which the panel normally drops because whether a deposition models them is a "
        "property of its refinement and not of the molecule -- so the ablation is the version a "
        "reasonable person would not think to question",
    )
    parser.add_argument("--n-particles", type=int, default=2000, help="pool size per marginal rung")
    parser.add_argument(
        "--shift-sigma",
        type=float,
        default=0.0,
        help="width in pixels of the translation each simulated particle is drawn with, and "
        "marginalized over. 0 draws every particle exactly centered, which no picked particle "
        "is; 1.6 gives the +-4 pixel lattice the real-data arms use.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    # `~` is expanded HERE, not by the shell: a quoted or job-submitted path arrives literal and would
    # otherwise make a directory named `~`. expanduser (a no-op without a tilde) covers every Path.
    for _name, _value in vars(args).items():
        if isinstance(_value, Path):
            setattr(args, _name, _value.expanduser())
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    # `only` goes IN, so a one-row job resolves one row's depositions and cannot die on another's
    candidates, unresolved = load_panel(args.panel, args.data_dir, fetch=args.fetch, only=args.only)
    for message in unresolved:
        # printed with the same "dropped:" prefix the correspondence refusals use, so one grep finds
        # every row a run declined to score whatever declined it
        print(f"dropped: {message}", flush=True)

    rows, rejections = rank_targets(
        candidates,
        plan,
        box_size=args.box_size,
        rmsf=args.rmsf,
        error_target=args.sigma,
        fraction=args.fraction,
        hydrogens=args.ablate == "hydrogens",
        form_factor="z" if args.ablate == "form-factor" else "peng",
        min_shared_fraction=args.min_shared_fraction,
        max_plausible_rmsd=None if args.no_rmsd_guard else _MAX_PLAUSIBLE_RMSD,
    )
    # the guards that ran, printed for the same reason `survey_panel` generates its note: a campaign
    # that relaxed one leaves rows a later reader cannot distinguish from rows that passed it
    print(
        f"correspondence guards: shared fraction >= "
        f"{args.min_shared_fraction if args.min_shared_fraction is not None else 0.75:g}, "
        f"Ca RMSD guard {'OFF' if args.no_rmsd_guard else f'{_MAX_PLAUSIBLE_RMSD:g} A'}",
        flush=True,
    )
    nulls: dict[str, dict] = {}

    def record(current: list[TargetRow]) -> str:
        """Write the result as it stands, and return the table.

        Called after EVERY marginal row rather than once at the end. The marginal arm runs for hours
        and can end in a kernel OOM kill with no traceback, so a run that dies partway should still be
        worth the rows it finished.
        """
        table = format_table(current, budget=args.budget)
        write_result(
            args.out,
            {
                "rows": [row.as_dict() for row in current],
                "rejections": rejections,
                "table": table,
                "n_scored": len(current),
                "n_rejected": len(rejections),
                "n_marginal_measured": sum(row.n_marginal is not None for row in current),
                # keyed by row name rather than positional: `rows` is re-sorted before the final
                # write, and a null that no longer lines up with its row is worse than none
                "nulls": nulls,
                "n_nulls_measured": sum(null.get("measured", False) for null in nulls.values()),
            },
            driver=DRIVER,
            arguments=vars(args),
        )
        return table

    if args.marginalize:
        by_name = {candidate.name: candidate for candidate in candidates}
        measured = []
        for position, row in enumerate(rows, start=1):
            # flushed, and BEFORE the work rather than after: an OOM kill names the row it died on
            # only if the row was announced on the way in
            print(f"[{position}/{len(rows)}] marginal rung: {row.name} ...", flush=True)
            started = time.monotonic()
            scored, null = measure_marginal(row, by_name[row.name], plan, args)
            measured.append(scored)
            nulls[row.name] = null
            print(f"[{position}/{len(rows)}] {row.name} done in {time.monotonic() - started:.0f}s", flush=True)
            record(measured)
        # a row whose chain run failed sorts LAST, not by its floor (which would put it atop the ranking);
        # ties break on pose efficiency only deterministically, its own order not being a real ranking
        rows = sorted(
            measured,
            key=lambda row: (
                row.n_marginal is None,
                row.n_marginal if row.n_marginal is not None else 0.0,
                -(row.pose_efficiency if row.pose_efficiency is not None else 0.0),
            ),
        )

    table = record(rows)
    print(table)
    for rejection in rejections:
        print(f"dropped: {rejection}")
    print(f"\n{len(rows)} scored, {len(rejections)} dropped -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
