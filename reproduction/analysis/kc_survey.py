"""Reading the convergent K_c survey: each pair's ladder plateau, its geometry, and the census.

Shared by the Eq 1 fit and the archived dose-covariate fit, which read the same records the same way.
Each converged pair's plateau is the crossover amplitude of its own d'^2(K) ladder; a pair is
admitted only if its two models share `MIN_OVERLAP` of their atoms.

Rationale: docs/rationale/reproduction/experiments/survey_law.md
"""

import glob
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from cryodisc.specimen.structures import _MIN_SHARED_FRACTION
from cryodisc.statistics.convergence import CROSSOVER_EXPONENT
from reproduction.targets import _molecular_mass, load_panel

A_EXP = CROSSOVER_EXPONENT
"""The crossover exponent of the convergence law `crossover` fits; derived in `CROSSOVER_EXPONENT`."""

REF_DOSE = 40.0
"""The single dose, in e/A^2, the reported survey is collected at."""

MIN_OVERLAP = _MIN_SHARED_FRACTION
"""The alignment's own default correspondence floor, fixed before the fit was seen rather than tuned."""


def crossover(k, amplitude, half):
    """The convergence law d'^2(K) = A / (1 + (K*/K)^p); returns d'^2 at bank size k."""
    return amplitude / (1.0 + (half / k) ** A_EXP)


def plateau(curve):
    """Extrapolated plateau A and half-recovery K* from one {K: d'^2} ladder, or None if unfittable."""
    pts = sorted((int(k), float(v)) for k, v in curve.items())
    if len(pts) < 3:
        return None
    k = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    try:
        (amp, half), _ = curve_fit(
            crossover,
            k,
            y,
            p0=[y.max() * 1.2, 8000.0],
            bounds=([y.max() * 0.9, 10], [y.max() * 6 + 1, 5e5]),
            maxfev=30000,
        )
    except RuntimeError:
        return None
    return amp, half, len(pts)


def free_exponent(curve):
    """The exponent one ladder prefers when it is freed: A, K* and p fitted together, in log d'^2.

    A check on the derived `A_EXP`, not an input to anything: every plateau is read at the derived
    exponent, and this only says how far each pair's own curve departs from it.

    Returns:
        The fitted exponent, or None for a ladder too short to identify three parameters.
    """
    from scipy.optimize import least_squares

    pts = sorted((int(k), float(v)) for k, v in curve.items() if float(v) > 0)
    if len(pts) < 5:
        return None
    log_k = np.log([p[0] for p in pts])
    log_y = np.log([p[1] for p in pts])

    def residual(theta):
        log_a, log_half, exponent = theta
        return log_a - np.log1p(np.exp(exponent * (log_half - log_k))) - log_y

    start = [log_y.max() + 0.05, float(np.median(log_k)), A_EXP]
    fit = least_squares(residual, start, bounds=([-20.0, -5.0, 0.2], [20.0, 30.0, 3.0]))
    return float(fit.x[2]) if fit.success else None


def lever_arms(rg, asphericity, acylindricity):
    """RMS distance of the particle from each principal axis, from its gyration-tensor scalars.

    The scalars are the normalized ones `pair_features.gyration` records; rotating by `theta` about
    axis i moves the particle by `theta * rho_i` in RMS, so these set how fast a rotation decorrelates it.
    """
    total = np.asarray(rg, float) ** 2
    rest = total * (1.0 - np.asarray(asphericity, float)) / 1.5  # the two minor eigenvalues together
    eigen = np.stack([total - rest, (rest + acylindricity * total) / 2, (rest - acylindricity * total) / 2], -1)
    return np.sqrt(np.clip(total[..., None] - eigen, 0.0, None))


def rms_frequency(power, weight, frequency):
    """RMS spatial frequency of the whitened difference, `power * weight` per shell, in `frequency` units."""
    whitened = np.asarray(power, float) * np.asarray(weight, float)
    return float(np.sqrt((whitened * np.asarray(frequency, float) ** 2).sum() / whitened.sum()))


def correlation_cells(k_rms, arms):
    """How many cells of SO(3) the difference decorrelates across: `8 pi^2` over the cell volume.

    About axis i the difference decorrelates over `sqrt(3) / (2 pi k_rms rho_i)` radians, the
    isotropic average of a phase `2 pi k . (omega x r)`; a bank resolves the difference once it
    samples cells of that size, so the half-recovery count is a fixed share of this number.
    """
    return 8 * np.pi**2 * np.prod(2 * np.pi * k_rms * np.asarray(arms, float) / np.sqrt(3.0), axis=-1)


def pool_index(patterns):
    """Map pair name -> (mass_kda, shared_fraction) from the survey-pool records that carry them."""
    index = {}
    for path in glob.glob(patterns):
        try:
            rows = json.loads(Path(path).read_text()).get("result", {}).get("rows", [])
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            name = row.get("name")
            if name and name not in index and row.get("mass_kda") and row.get("shared_fraction") is not None:
                index[name] = (row["mass_kda"], row["shared_fraction"])
    return index


def candidate_index(panels):
    """Map pair name -> its panel candidate, so a pool-missing pair costs one lookup, not a re-glob."""
    index = {}
    for pattern in panels:
        for panel in glob.glob(pattern):
            try:
                candidates, _ = load_panel(panel, data_dir="data/pdb", fetch=False)
            except Exception:  # noqa: BLE001, S112 -- an unreadable panel only loses its own names
                continue
            for candidate in candidates:
                name = getattr(candidate, "name", None)
                if name and name not in index:  # first panel wins, as the per-name scan did
                    index[name] = candidate
    return index


def geometry(names, pool_glob, panel_globs):
    """Map pair name -> (mass_kda, shared_fraction) for every name, from the pool or its alignment.

    A pool-missing pair is re-aligned at no overlap floor, as the survey scored it, so its overlap is
    measured rather than refused; the floor is applied once, by the caller, to every pair alike.
    """
    known = pool_index(pool_glob)
    missing = [name for name in names if name not in known]
    if missing:
        panels = candidate_index(panel_globs)
        for name in missing:
            if name not in panels:
                raise SystemExit(f"{name!r} converged but is in no panel under {panel_globs}")
            pair = panels[name].align(min_shared_fraction=0.0)
            known[name] = (_molecular_mass(pair), pair.alignment.shared_fraction)
    return {name: known[name] for name in names}


def load_survey(survey_glob):
    """Every survey record (result payload), read once for both the census and the per-pair fit."""
    records = []
    for path in glob.glob(survey_glob):
        if "_tmp" in path:
            continue
        record = json.loads(Path(path).read_text())
        records.append(record.get("result", record))
    return records


def census(records):
    """(attempted, converged) unique pairs across the survey, the funnel the Methods quotes."""
    attempted, converged = set(), set()
    for record in records:
        name = record.get("name")
        if not name:
            continue
        attempted.add(name)
        if record.get("status") == "converged":
            converged.add(name)
    return len(attempted), len(converged)


def gather(records, admitted):
    """Per-pair (mass, rmsd, dose, plateau amplitude, ladder length) for every admitted converged pair."""
    rows = []
    for record in records:
        name, rmsd, dose = record.get("name"), record.get("rmsd"), record.get("dose")
        if record.get("status") != "converged" or name not in admitted or not rmsd or not dose or rmsd <= 0:
            continue
        fitted = plateau(record.get("curve") or {})
        if fitted is None:
            continue
        rows.append(
            {
                "name": name,
                "mass_kda": admitted[name][0],
                "shared_fraction": admitted[name][1],
                "ca_rmsd": rmsd,
                "dose": dose,
                "d2_plateau": fitted[0],
                "kc": fitted[1],
                "n_rungs": fitted[2],
            }
        )
    return rows
