"""One straight line, fitted once, for everything in this package that fits one.

`line_fit` is the shared estimator: `fit_purity` regresses a measured fraction on a known one, and
panel drivers regress in LOGS to read an exponent. `weighted_power_law` and `log_curvature` are the
WEIGHTED pair, for rungs that carry an error bar. Consolidated so one statistic keeps a single
spelling across result files, which lets two surveys merge without a human reconciling column names.
"""

import math

import numpy as np


def line_fit(abscissa: np.ndarray, ordinate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares of a straight line, with the covariance the residuals imply.

    The residual variance `r.r / (n - 2)` is zero at `n = 2` and reported as an error of exactly
    zero rather than NaN, which is why callers refuse fewer than three points.

    Args:
        abscissa: The known quantity, one per point. Already centered however the caller wants it.
        ordinate: The measured quantity, in the same order.

    Returns:
        `(coefficients, covariance, residual_rms)`, with `coefficients` as `(intercept, slope)` and
        `covariance` its 2x2 matrix. `residual_rms` divides by `n`, not the degrees of freedom, and
        is read as a check that the model IS a line.
    """
    design = np.stack([np.ones_like(abscissa), abscissa], 1)
    coefficients, *_ = np.linalg.lstsq(design, ordinate, rcond=None)
    residual = ordinate - design @ coefficients
    dof = abscissa.size - 2
    variance = float(residual @ residual) / dof if dof > 0 else 0.0
    covariance = variance * np.linalg.inv(design.T @ design)
    return coefficients, covariance, float(np.sqrt(residual @ residual / abscissa.size))


def recovery_error(p_true: np.ndarray | list[float], p_hat: np.ndarray | list[float]) -> dict:
    """Recovery accuracy: the RMS error in percentage points, with its calibration decomposition.

    The headline for a fraction-recovery sweep is ACCURACY -- how far the recovered fraction is from
    the truth -- not the slope alone, which is only the compression and misses an offset. `rms_pp` is
    the accuracy; `slope` and `intercept` regress `p_hat` on `p_true` so a reader sees WHERE the error
    is (a compression shows in the slope below one, a directional offset in the intercept away from zero).

    Args:
        p_true: Known state-A fraction of each mixture.
        p_hat: Recovered fraction of each, in the same order.

    Returns:
        `{"rms_pp", "slope", "intercept"}` -- the RMS of `p_hat - p_true` in percentage points, and the
        `(intercept, slope)` of `p_hat` regressed on `p_true`.
    """
    true = np.asarray(p_true, float).ravel()
    hat = np.asarray(p_hat, float).ravel()
    rms_pp = 100.0 * float(np.sqrt(np.mean((hat - true) ** 2)))
    (intercept, slope), _covariance, _residual = line_fit(true, hat)
    return {"rms_pp": rms_pp, "slope": float(slope), "intercept": float(intercept)}


def fit_power_law(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict:
    """`y = a x^b` by least squares in logs, with the exponent's standard error.

    One set of key names, so `exponent` never appears as `slope` in another result file and two
    surveys merge without reconciliation.

    Args:
        x: Positive predictor values. Non-positive rows are dropped, since a log of one is not a
            number and silently keeping it would poison the fit.
        y: Positive response values, in the same order.

    Returns:
        `n` (rows that survived), and where a fit was possible `exponent`, its standard `error`,
        the `intercept` in logs, and the Pearson `r` in logs. Only `n` when fewer than three rows
        survive -- a line through two points has no residual to take an error from -- or when every
        surviving abscissa is the same, where a slope is a division by zero dressed as an exponent.
    """
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if a > 0 and b > 0]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2:
        return {"n": len(pairs)}
    log_x = np.log(np.array([a for a, _ in pairs]))
    log_y = np.log(np.array([b for _, b in pairs]))
    (intercept, slope), covariance, _ = line_fit(log_x, log_y)
    return {
        "n": len(pairs),
        "exponent": float(slope),
        "error": float(math.sqrt(max(covariance[1, 1], 0.0))),
        "intercept": float(intercept),
        "r": float(np.corrcoef(log_x, log_y)[0, 1]),
    }


def _goodness(residuals: np.ndarray, weights: np.ndarray, dof: int) -> dict:
    """Weighted chi-square of a log-space fit, and the range it was earned over.

    The slope's standard error alone cannot say whether a power law is the right shape: it shrinks
    with the fitted range whether or not the points lie on a line, so a wide window can report a
    tight exponent that no straight line passes through. chi2/dof is what separates the two, and it
    is what the fit range is chosen on.

    Args:
        residuals: `log y` minus the fit, at each rung.
        weights: `1 / var(log y)`, the same weights the fit used.
        dof: Rungs minus fitted parameters.

    Returns:
        The chi-square, its degrees of freedom, and their ratio (`None` when the fit is saturated).
    """
    chi_square = float(np.sum(weights * residuals**2))
    return {
        "chi_square": chi_square,
        "dof": int(dof),
        "chi_square_per_dof": float(chi_square / dof) if dof > 0 else None,
    }


def weighted_power_law(n: np.ndarray, y: np.ndarray, sigma: np.ndarray, slope: float | None = None) -> dict:
    """Weighted least squares of `log y` on `log n`, optionally with the slope fixed.

    Args:
        n: Rung sizes.
        y: The signal at each rung, strictly positive.
        sigma: Standard error on `y`, propagated to logs as `sigma / y`.
        slope: If given, fix the exponent and fit only the amplitude.

    Returns:
        The fitted parameters with their standard errors.

    Raises:
        SystemExit: If fewer than two rungs survive, or any signal is non-positive.
    """
    if np.any(y <= 0):
        raise SystemExit(
            f"{int((y <= 0).sum())} of {y.size} rungs have a non-positive signal and the fit is in "
            "logs. A memorization term that changed sign is a finding, not a fit."
        )
    weights = (y / sigma) ** 2  # 1 / var(log y)
    log_n, log_y = np.log(n), np.log(y)
    if slope is None:
        if n.size < 3:
            raise SystemExit(f"a free-slope fit needs three rungs, got {n.size}")
        design = np.stack([log_n, np.ones_like(log_n)], axis=1)
        weighted = design * np.sqrt(weights)[:, None]
        solution, *_ = np.linalg.lstsq(weighted, log_y * np.sqrt(weights), rcond=None)
        covariance = np.linalg.inv(weighted.T @ weighted)
        residuals = log_y - design @ solution
        return {
            "slope": float(solution[0]),
            "slope_sem": float(math.sqrt(covariance[0, 0])),
            "log_amplitude": float(solution[1]),
            "amplitude": float(math.exp(solution[1])),
            "amplitude_sem": float(math.exp(solution[1]) * math.sqrt(covariance[1, 1])),
            **_goodness(residuals, weights, n.size - 2),
        }
    if n.size < 2:
        raise SystemExit(f"a fixed-slope fit needs two rungs, got {n.size}")
    residual = log_y - slope * log_n
    intercept = float(np.sum(weights * residual) / np.sum(weights))
    variance = float(1.0 / np.sum(weights))
    return {
        "slope": float(slope),
        "slope_sem": 0.0,
        "log_amplitude": intercept,
        "amplitude": float(math.exp(intercept)),
        "amplitude_sem": float(math.exp(intercept) * math.sqrt(variance)),
        **_goodness(residual - intercept, weights, n.size - 1),
    }


def log_curvature(n: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict | None:
    """Fit `log y` on `log n` with a quadratic term, and report how far from zero it lands.

    A power law that has not reached its asymptote curves in logs, and the free-slope fit absorbs
    that curvature into the exponent rather than reporting it -- which is how a misspecified fit
    returns a precise, wrong slope. This is the direct test: a curvature consistent with zero is
    what earns the right to quote a single exponent over the window.

    Args:
        n: Rung sizes.
        y: The signal at each rung, strictly positive.
        sigma: Standard error on `y`.

    Returns:
        The quadratic coefficient with its standard error and significance, or `None` when there
        are too few rungs to fit three parameters and have anything left over.
    """
    if n.size < 4:
        return None
    weights = (y / sigma) ** 2
    log_n = np.log(n)
    # centered, so the coefficient does not trade against the intercept over a window far from n = 1
    centered = log_n - log_n.mean()
    design = np.stack([centered**2, centered, np.ones_like(centered)], axis=1)
    weighted = design * np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(weighted, np.log(y) * np.sqrt(weights), rcond=None)
    covariance = np.linalg.inv(weighted.T @ weighted)
    error = math.sqrt(covariance[0, 0])
    return {
        "curvature": float(solution[0]),
        "curvature_sem": float(error),
        "sigma": float(abs(solution[0]) / error) if error > 0 else None,
    }
