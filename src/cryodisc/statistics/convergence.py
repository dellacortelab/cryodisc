"""The marginal rung at an infinitely fine bank, extrapolated in bank fineness `K`.

`pose_log_ratio` marginalizes over a finite bank and returns a log-ratio too small, so the integral is
approached FROM BELOW, with a deficit falling as `K^(-p)`, `p = CROSSOVER_EXPONENT`, so a straight
line through `K^(-p)` extrapolates it. The exponent is ASYMPTOTIC and describes a HAAR pool; fit only over Haar, over a wide enough span in `K`. Use three grids, never two:
two determine a line exactly, so the reported error would be exactly 0.

Rationale: docs/rationale/statistics/convergence.md
"""

from collections.abc import Callable, Sequence

CROSSOVER_EXPONENT = 1.0
"""`p` in the bank-convergence deficit `d'^2 - d'^2(K) ~ K^(-p)`, and in `d'^2(K) = d'^2/(1 + (K*/K)^p)`.

The deficit is the discriminability carried by angular modes finer than a K-node bank resolves. K
near-uniform orientations on a d-dimensional manifold resolve modes to degree L ~ K^(1/d); a difference
spectrum falling as l^(-gamma) leaves a deficit ~ L^(-(gamma-1)) ~ K^(-(gamma-1)/d). A density step across
the molecular surface, a jump over a (d-1)-surface, has gamma = d + 1, so p = 1 whatever d is. 2/3 is the
trapezoid rule's worst-case bound on a 3-manifold, not the rate a surface-dominated difference attains.
Rationale: docs/rationale/statistics/convergence.md"""
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

_Artifact = TypeVar("_Artifact")

# The pose-bank ladder every marginal grows up, and how convergence is judged. Lifted from
# `reproduction/experiments/panel/kc_survey.py` so that the survey and every other experiment converge by ONE rule
# rather than each carrying a fixed `--nodes`. Requested sizes; `pose_bank()` snaps each to the nearest
# balanced Hopf grid, so the realized count is read back and keyed on.
DEFAULT_LADDER: tuple[int, ...] = (504, 2016, 8064, 33384, 64128, 128000, 256000, 600000, 1_000_000)
RISE = 0.15
"""Top-octave d^2 gain at or above which the ladder is still rising and the bank must grow further."""
INITIAL = 3
"""Banks scored before adaptive stepping begins -- three is the minimum `extrapolate_in_k` will fit."""


class RefusedExtrapolation(ValueError):
    """A limit `Extrapolation.check` refused, carrying the ladder that produced it.

    A `ValueError` so every caller that already catches one keeps working, and every docstring
    promising one stays true. It carries the fit because the counts and values are what say WHY a
    limit was refused, and a refusal that discards them leaves nothing to act on.

    Args:
        message: Why the limit was refused, verbatim from the test that refused it.
        extrapolation: The fit, points included.
    """

    def __init__(self, message: str, extrapolation: "Extrapolation") -> None:
        super().__init__(message)
        self.extrapolation = extrapolation


@dataclass(frozen=True)
class Extrapolation:
    """A rung extrapolated to an infinitely fine bank, and what qualifies it.

    Args:
        limit: `d^2` at `K -> infinity`, the intercept of the weighted straight-line fit.
        limit_err: Standard error of the intercept, inflated by the fit's own scatter when the
            points sit worse on a line than their error bars claim. Never deflated: a fit that
            agrees better than its error bars is not evidence that the bars were pessimistic.
        slope: `c` in `d^2(K) = limit - c K^(-p)`, `p = CROSSOVER_EXPONENT`. Positive is the expected sign, the ladder
            rising toward its limit as the bank refines.
        counts: Node counts that were fitted, ascending.
        values: The `d^2` measured at each.
        chi2_per_dof: Goodness of the straight line. Far above one means the assumed exponent is not
            the law these points follow, and the intercept is an extrapolation of the wrong curve.
            Dimensionless ONLY when real errors were supplied -- see `weighted`.
        exponent: The `p` in `K^(-p)` that was fitted against, `CROSSOVER_EXPONENT` unless a caller measured another.
        weighted: Whether real error bars were given. Without them `chi2_per_dof` carries the fit's own
            scatter as its scale, so it estimates the error rather than testing the law, and `check`
            cannot use it.
        errors: The standard error of each value, in the same order, or empty when none were supplied.
            Carried so the fit is REPRODUCIBLE from what it reports: the weights it derived from these
            are not recoverable from the weights, and `validation.deficit` needs the bars to correct a
            ladder that this fit refused.
        local_exponents: The exponent each consecutive pair implies, given this fit's limit:
            `-log((limit - y2)/(limit - y1)) / log(K2/K1)`. All equal to `exponent` when the law holds,
            drifting when the banks are short of the asymptote. A diagnostic and NOT a guard -- see the
            module docstring for the span below which its noise exceeds its signal. NaN where a point
            sits at or beyond the limit, which `check` catches separately.
    """

    limit: float
    limit_err: float
    slope: float
    counts: tuple[int, ...]
    values: tuple[float, ...]
    chi2_per_dof: float
    exponent: float = CROSSOVER_EXPONENT
    local_exponents: tuple[float, ...] = ()
    errors: tuple[float, ...] = ()

    @property
    def weighted(self) -> bool:
        """Whether real error bars went into the fit, which is whether any were supplied.

        `chi2_per_dof` is dimensionless only when this is true, which is why the goodness-of-fit test
        is gated on it.
        """
        return bool(self.errors)

    @property
    def span(self) -> float:
        """`K_max / K_min`, the lever arm the exponent has to be identified from.

        Returns:
            The ratio. See the module docstring for the sweep that sets what it has to exceed: the
            intercept's bias under a wrong exponent is invisible to every residual-based test below
            roughly 20x and unmissable at 64x.
        """
        return max(self.counts) / min(self.counts) if self.counts else 1.0

    def check(
        self,
        *,
        ceiling: float | None = None,
        max_chi2_per_dof: float = 4.0,
        min_span: float = 64.0,
    ) -> None:
        """Refuse a limit that is impossible, raising an error that carries the ladder.

        A thin wrapper over `_refuse`, which holds every actual test. It exists so a refusal does not
        destroy its own evidence: the raised error carries the fit so a caller that catches it still
        has the points behind the message.

        Args:
            ceiling: As `_refuse`.
            max_chi2_per_dof: As `_refuse`.
            min_span: As `_refuse`.

        Raises:
            RefusedExtrapolation: With the same message `_refuse` raises, and this fit attached. A
                `ValueError`, so a caller that already catches one is unaffected.
        """
        try:
            self._refuse(ceiling=ceiling, max_chi2_per_dof=max_chi2_per_dof, min_span=min_span)
        except ValueError as error:
            raise RefusedExtrapolation(str(error), self) from error

    def _refuse(
        self,
        *,
        ceiling: float | None = None,
        max_chi2_per_dof: float = 4.0,
        min_span: float = 64.0,
    ) -> None:
        """Refuse a limit that the ladder it came from, or its own goodness of fit, says is impossible.

        The extrapolation is ordinary least squares and knows nothing about what it is fitting, so
        every constraint it must obey has to be imposed here.

        Args:
            ceiling: The `known_pose` rung, if known. The marginal rung cannot exceed it: knowing
                the true pose exactly must beat marginalizing it against any prior.

                **It must be `known_pose`, and no bank-dependent quantity.** `limit` is the marginal
                rung extrapolated to K -> infinity, so a ceiling measured on one bank is not an
                inequality anyone is entitled to.
            max_chi2_per_dof: How badly the points may sit on the assumed power law before the limit is
                refused. Only enforced when real errors were supplied, `chi2` being unitless otherwise.
                Four is two sigma per point at one degree of freedom, loose because the alternative to a
                bad extrapolation is no rung at all.
            min_span: Least `K_max / K_min` the grids may span, the default being where the module
                docstring's sweep found a wrong exponent unmissable. Pass `0.0` to accept a narrow span
                deliberately and quote the limit as indicative rather than measured.

        Raises:
            ValueError: If the limit is not positive, sits on the wrong side of the measured points,
                violates either bound it was given, or the points do not follow the assumed exponent.
        """
        if not np.isfinite(self.limit):
            raise ValueError(f"the extrapolated limit is {self.limit}, so the fit failed outright")
        # FIRST, because it is the only failure here that no amount of data quality can rescue
        if self.span < min_span:
            raise ValueError(
                f"the grids {list(self.counts)} span only {self.span:.3g}x in K, against a required "
                f"{min_span:.3g}x, so the exponent is NOT IDENTIFIABLE: assuming K^(-{self.exponent:.4g}) "
                f"when the truth is 0.55 biases the intercept 6 sigma at a chi2/dof of 2.1. Neither this "
                f"fit's chi2 ({self.chi2_per_dof:.4g}) nor its exponents "
                f"{[round(p, 3) for p in self.local_exponents]} can tell you whether {self.limit:.4g} is "
                "right. Add a coarser or a much finer bank, or pass min_span=0.0 to quote it as indicative."
            )
        if self.limit <= 0.0:
            raise ValueError(
                f"the extrapolated limit is {self.limit:.4g}, which is not a squared distance. The "
                f"points {self.values} do not lie on a line in K^(-p) that reaches a positive "
                "intercept; add grids rather than quoting this."
            )
        # which side is correct depends on the direction the measured ladder runs, and a ladder that
        # FALLS in K is real -- a finer bank can move a particle's posterior off a node it landed on
        # by luck. So the limit must lie beyond the last point in whichever direction the fit found.
        finest = self.values[-1]
        if self.slope > 0.0 and self.limit < finest:
            raise ValueError(
                f"the ladder rises in K (slope {self.slope:.4g}) but the limit {self.limit:.4g} sits "
                f"BELOW its finest grid, {finest:.4g}. That is arithmetic, not a rung."
            )
        if self.slope < 0.0 and self.limit > finest:
            raise ValueError(
                f"the ladder falls in K (slope {self.slope:.4g}) but the limit {self.limit:.4g} sits "
                f"ABOVE its finest grid, {finest:.4g}. That is arithmetic, not a rung."
            )
        # a NON-FINITE local exponent means the limit sits BETWEEN the points, so `limit - y` changes
        # sign across the ladder and its logarithm is undefined. The two checks above cannot see it:
        # each tests the limit against the finest point alone, so a non-monotone ladder can slip past.
        if self.local_exponents and not np.all(np.isfinite(self.local_exponents)):
            raise ValueError(
                f"the limit {self.limit:.4g} sits between the points it was fitted to, {self.values}, so "
                f"the pairwise exponents are {[round(p, 3) for p in self.local_exponents]} rather than "
                "numbers. A ladder that does not approach its limit from ONE side is not converging in K "
                "at all, and no intercept through it is a rung -- whatever its chi2 says, since a "
                "straight line can fit a non-monotone ladder perfectly well. Add grids, or find out why "
                "the rung moves both ways."
            )
        if ceiling is not None and self.limit > ceiling:
            raise ValueError(
                f"the extrapolated marginal rung {self.limit:.4g} exceeds the known-pose rung "
                f"{ceiling:.4g}. Marginalizing a pose against its prior cannot beat scoring at the "
                "true pose, so one of the two is wrong."
            )
        # last, because the bounds above name a wrong number and this names a wrong CURVE
        if self.weighted and self.chi2_per_dof > max_chi2_per_dof:
            drift = ", ".join(f"{p:.3g}" for p in self.local_exponents)
            raise ValueError(
                f"the points do not follow K^(-{self.exponent:.4g}): chi2/dof is {self.chi2_per_dof:.4g} "
                f"against a tolerance of {max_chi2_per_dof:.4g}, so the intercept {self.limit:.4g} "
                f"extrapolates the wrong curve and its error bar {self.limit_err:.4g} does not cover being "
                f"wrong about the exponent. The pairwise exponents are [{drift}]; drifting upward means a "
                "bank short of the asymptote. Add finer grids, or pass the exponent you measured."
            )


def extrapolate_in_k(
    counts: Sequence[int],
    values: Sequence[float],
    errors: Sequence[float] | None = None,
    exponent: float = CROSSOVER_EXPONENT,
) -> Extrapolation:
    """Fit `d^2(K) = limit - c K^(-p)` and return the intercept.

    Args:
        counts: Node count of each bank the rung was measured on. At least three DISTINCT values --
            see the module docstring for why two is not a fit.
        values: The rung measured on each, in the same order.
        errors: Standard error of each, as `chain.Rung.d2_err` reports. `None` weights the points
            equally, which is right only when the pools were the same size. It also disables `check`'s
            goodness-of-fit refusal, `chi2` having no scale to be judged against without them.
        exponent: The `p` in `K^(-p)`. The default is the asymptotic quadrature rate; override it with
            a MEASURED one when the banks are too coarse for that to hold, which the module docstring
            gives two datasets' worth of evidence for.

    Returns:
        The extrapolation. Call `check` on it before quoting it.

    Raises:
        ValueError: If fewer than three distinct counts are given, if the inputs disagree in length,
            if any count is not positive, if any error is negative, or if `exponent` is not positive.
    """
    if not exponent > 0.0:
        raise ValueError(f"the exponent in K^(-p) must be positive, got {exponent:.6g}")
    node_counts = np.asarray(counts, dtype=np.int64)
    measured = np.asarray(values, dtype=np.float64)
    if node_counts.shape != measured.shape:
        raise ValueError(f"counts and values must agree, got {node_counts.shape} and {measured.shape}")
    if np.unique(node_counts).size < 3:
        raise ValueError(
            f"need at least three distinct node counts, got {sorted(set(node_counts.tolist()))}. Two "
            "points determine a line exactly, so the residual and the degrees of freedom are both "
            "zero and the reported error would be 0 by construction rather than small."
        )
    if np.any(node_counts <= 0):
        raise ValueError(f"every node count must be positive, got {node_counts.tolist()}")

    if errors is None:
        weight = np.ones_like(measured)
    else:
        spread = np.asarray(errors, dtype=np.float64)
        if spread.shape != measured.shape:
            raise ValueError(f"errors must be one per value, got {spread.shape} and {measured.shape}")
        if np.any(spread < 0.0):
            raise ValueError(f"errors must be non-negative, got {spread.tolist()}")
        # a rung computed from the model alone carries error 0, which as a weight is infinite and
        # would pin the line through that point; the floor makes it merely dominant
        floor = np.max(spread) * 1e-3 if np.max(spread) > 0.0 else 1.0
        weight = 1.0 / np.maximum(spread, floor) ** 2

    order = np.argsort(node_counts)
    node_counts, measured, weight = node_counts[order], measured[order], weight[order]
    abscissa = node_counts.astype(np.float64) ** (-exponent)

    total = weight.sum()
    mean_x = (weight * abscissa).sum() / total
    mean_y = (weight * measured).sum() / total
    variance_x = (weight * (abscissa - mean_x) ** 2).sum()
    if variance_x <= 0.0:
        raise ValueError(f"every grid has the same K^(-{exponent:.4g}); there is no line to fit")
    covariance = (weight * (abscissa - mean_x) * (measured - mean_y)).sum()

    # d^2 = intercept + gradient * K^(-p), and the module writes the law as `limit - c K^(-p)`,
    # so the reported slope is the negated gradient and is positive for a ladder that rises in K
    gradient = covariance / variance_x
    intercept = mean_y - gradient * mean_x

    residual = measured - (intercept + gradient * abscissa)
    dof = measured.size - 2
    chi2 = float((weight * residual**2).sum())
    # with real error bars the weights carry the scale, so only chi2/dof ABOVE one inflates; without
    # them the weights are unitless and chi2/dof IS the variance estimate, so it is taken as it comes
    scale = chi2 / dof if errors is None else max(chi2 / dof, 1.0)
    variance_intercept = (1.0 / total + mean_x**2 / variance_x) * scale

    # the exponent each consecutive PAIR implies given this intercept, all equal to `exponent` when
    # the law holds. `errstate` because a point on the far side of the limit gives a non-positive
    # deficit, whose log is nan by design; `check` refuses that case on the limit itself.
    with np.errstate(divide="ignore", invalid="ignore"):
        # take the deficit on the side the ladder approaches from (what `gradient` names), so a
        # falling ladder is not misread as straddling its limit
        deficit = (intercept - measured) if gradient < 0.0 else (measured - intercept)
        local = -np.diff(np.log(deficit)) / np.diff(np.log(node_counts.astype(np.float64)))

    return Extrapolation(
        limit=float(intercept),
        limit_err=float(np.sqrt(variance_intercept)),
        slope=float(-gradient),
        counts=tuple(int(k) for k in node_counts),
        values=tuple(float(v) for v in measured),
        chi2_per_dof=float(chi2 / dof) if dof > 0 else 0.0,
        exponent=float(exponent),
        # reordered with the points, so the triple stays aligned however the caller passed it
        errors=() if errors is None else tuple(float(e) for e in np.asarray(errors, dtype=np.float64)[order]),
        local_exponents=tuple(float(p) for p in local),
    )


def top_octave_gain(curve: dict[int, float]) -> float:
    """Fractional `d^2` gain across the top octave of a `{realized_bank: d2}` curve.

    The stopping signal: the finest bank against the one nearest half its size, as a fraction of the
    finest. Small means the ladder has flattened; `>= RISE` means it is still climbing and the bank
    must grow. One point (no octave yet) reads as fully rising.

    Args:
        curve: Measured `d^2` keyed by the realized node count of each bank.

    Returns:
        `(d2[top] - d2[half]) / d2[top]`, or `1.0` with fewer than two banks.
    """
    banks = np.array(sorted(curve))
    if banks.size < 2:
        return 1.0
    d2 = np.array([curve[int(b)] for b in banks])
    half = int(np.argmin(np.abs(banks - banks[-1] / 2)))
    return float((d2[-1] - d2[half]) / max(d2[-1], 1e-12))


@dataclass(frozen=True)
class Convergence:
    """The result of growing a pose bank until a grid-marginalized quantity stops rising.

    Args:
        curve: `{realized_bank: value}`, every rung scored, ascending in bank size.
        errors: Standard error at each bank in `curve`'s key order, or empty when the scorer gave none.
        plateau: The `K -> infinity` value. The checked `extrapolate_in_k` limit when the curve admits
            one, else the finest measured bank (`max(curve.values())`) as a censored lower bound.
        extrapolation: The `K^(-p)` fit behind `plateau`, or `None` when too few/failing points left
            the plateau at the finest bank instead.
        status: `"converged"` (the top octave gained less than `RISE`) or `"censored"` (still rising at
            the compute cap, so `plateau` is a lower bound, not the limit).
        top_octave_gain: The final rising signal, for the record and for a reader to discount the tail.
    """

    curve: dict[int, float]
    errors: tuple[float, ...]
    plateau: float
    extrapolation: Extrapolation | None
    status: str
    top_octave_gain: float

    def as_record(self) -> dict:
        """A JSON-ready record: curve keyed by string, plateau, status, gain, and realized node count."""
        return {
            "curve": {str(k): v for k, v in self.curve.items()},
            "plateau": self.plateau,
            "status": self.status,
            "top_octave_gain": self.top_octave_gain,
            "nodes_realized": max(self.curve),
        }


def converge_marginal(
    score_bank: Callable[[int], tuple[int, float] | tuple[int, float, float]],
    *,
    ladder: Sequence[int] = DEFAULT_LADDER,
    cap: int = 1_000_000,
    rise: float = RISE,
) -> Convergence:
    """Grow the pose bank up `ladder` until a grid-marginalized quantity converges, or the cap censors it.

    The one convergence rule for every experiment: rather than fixing `--nodes`, step up the bank ladder
    scoring each rung, stop when the top octave gains less than `rise` (`kc_survey`'s criterion), and read
    the plateau off the rigorous `K^(-p)` extrapolation the finished curve admits. A quantity still
    rising at `cap` is recorded censored with its finest bank as a lower bound, never chased.

    Args:
        score_bank: Scores ONE bank. Given a requested node count, returns `(realized_count, value)` or
            `(realized_count, value, error)` -- realized because `pose_bank` quantizes the request. The
            error, when given, weights the extrapolation and gates its goodness-of-fit refusal.
        ladder: Requested bank sizes, ascending; the default is the survey's. Sizes above `cap` are
            dropped so the cap, not the ladder, bounds the growth.
        cap: Largest bank to score. Reaching it while still rising is a censored result.
        rise: Top-octave gain below which the ladder counts as converged.

    Returns:
        A `Convergence`: the measured curve, the plateau (extrapolated when the curve allows, else the
        finest bank), and whether it converged or was censored at the cap.

    Raises:
        ValueError: If `ladder` has no size at or below `cap`.
    """
    rungs = sorted({int(k) for k in ladder if int(k) <= cap})
    if not rungs:
        raise ValueError(f"no bank in {list(ladder)} is at or below the cap {cap}")

    curve: dict[int, float] = {}
    errs: dict[int, float] = {}
    for step, requested in enumerate(rungs):
        scored = score_bank(requested)
        realized, value = int(scored[0]), float(scored[1])
        curve[realized] = value
        if len(scored) > 2:
            errs[realized] = float(scored[2])
        gain = top_octave_gain(curve)
        # keep growing through the cheap INITIAL prefix before the octave test can mean anything, then
        # stop the first time the top octave has flattened
        if step + 1 >= INITIAL and gain < rise:
            return _finish(curve, errs, "converged", gain)
    return _finish(curve, errs, "censored", top_octave_gain(curve))


def _finish(curve: dict[int, float], errs: dict[int, float], status: str, gain: float) -> Convergence:
    """Attach the extrapolated plateau to a finished curve, falling back to the finest bank."""
    order = sorted(curve)
    errors = tuple(errs[k] for k in order) if len(errs) == len(curve) else ()
    extrapolation = None
    plateau = max(curve.values())
    if len({round(k) for k in curve}) >= 3:
        try:
            fit = extrapolate_in_k(order, [curve[k] for k in order], errors=errors or None)
            fit.check(min_span=0.0)  # the ladder's own span is enormous; guard sign/monotonicity, not span
            extrapolation, plateau = fit, fit.limit
        except (ValueError, RefusedExtrapolation):
            extrapolation, plateau = None, max(curve.values())
    return Convergence(dict(sorted(curve.items())), errors, float(plateau), extrapolation, status, float(gain))


def converge_caching(
    simulate: Callable[[int], tuple[int, _Artifact]],
    value: Callable[[_Artifact], float],
    *,
    ladder: Sequence[int] = DEFAULT_LADDER,
    cap: int = 1_000_000,
    rise: float = RISE,
) -> tuple[Convergence, _Artifact]:
    """Converge `value` over the pose ladder while keeping the finest bank's artifact.

    `converge_marginal` returns only the scalar curve, but a consumer that needs the full per-bank
    object -- the per-particle Lambda arrays, say -- at the converged grid cannot rebuild it from that.
    Here `simulate(k)` returns the realized count and that object, `value` reads the scalar to converge
    off it (d^2), and the object at the finest scored bank is returned beside the `Convergence`.

    Only that one artifact is held, not every bank's: the finest is the sole one any caller reads, and
    each carries a bank's worth of per-particle arrays, so keeping all of them would cost the ladder's
    whole memory for nothing.

    Args:
        simulate: Builds one bank's artifact, returning `(realized_count, artifact)`.
        value: The grid-marginalized scalar to converge, read off the artifact.
        ladder: Requested bank sizes, ascending; passed through to `converge_marginal`.
        cap: Largest bank to score; the finest scored bank's artifact is the one returned.
        rise: Top-octave gain below which the ladder counts as converged.

    Returns:
        The `Convergence`, and the artifact at the finest scored bank.
    """
    finest_count = -1
    finest_artifact: _Artifact | None = None

    def score_bank(requested: int) -> tuple[int, float]:
        nonlocal finest_count, finest_artifact
        realized, artifact = simulate(requested)
        if realized >= finest_count:  # >= so a tie keeps the later one, as the old dict did
            finest_count, finest_artifact = realized, artifact
        return realized, float(value(artifact))

    convergence = converge_marginal(score_bank, ladder=ladder, cap=cap, rise=rise)
    return convergence, finest_artifact
