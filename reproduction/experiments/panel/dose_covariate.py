"""Freeze the retired dose covariate from the archived multidose survey.

    .venv/bin/python reproduction/experiments/panel/dose_covariate.py

Dose enters the theory through the 1/D shot-noise term and was fit once on a randomized 20-80 e/A^2
design. The covariate was weak, so the reported survey fixes dose at the reference 40 e/A^2 and fits no
dose term (Eq 1). This freezes that one number as the appendix's evidence; the multidose survey is
archived and this driver is deliberately not part of the build.
"""

import argparse

import numpy as np
from scipy.optimize import curve_fit

from reproduction.analysis.kc_survey import MIN_OVERLAP, REF_DOSE, census, gather, geometry, load_survey
from reproduction.record import write_result

DRIVER = "reproduction/experiments/panel/dose_covariate.py"


def main() -> int:
    """Fit the dose covariate on the archived multidose survey and record it with provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", default="archive/kc_survey_multidose_2026-09-20/*.json")
    parser.add_argument("--pool", default="results/survey_pool/*.json")
    parser.add_argument("--panels", nargs="+", default=["data/survey_*.json"])
    parser.add_argument("--out", default="results/dose_covariate_archived.json")
    args = parser.parse_args()

    records = load_survey(args.survey)
    n_attempted, n_converged = census(records)
    converged = sorted({r["name"] for r in records if r.get("status") == "converged" and r.get("name")})
    # the same overlap floor Eq 1 admits by, so the covariate is fit on comparable d'^2
    admitted = {
        name: shape for name, shape in geometry(converged, args.pool, args.panels).items() if shape[1] >= MIN_OVERLAP
    }
    rows = gather(records, admitted)
    mass = np.array([r["mass_kda"] for r in rows])
    rmsd = np.array([r["ca_rmsd"] for r in rows])
    dose = np.array([r["dose"] for r in rows])
    amp = np.log([r["d2_plateau"] for r in rows])
    lnM, lnR, lnD, lnDc = np.log(mass), np.log(rmsd), np.log(dose), np.log(dose / REF_DOSE)

    # the same native-dose form eq1 once fit: mass^1, rmsd^2 fixed, dose as one shared covariate
    def native(_, lnC, ln_rstar, q, c):
        return lnC + lnM + 2 * lnR - np.log(1 + (rmsd / np.exp(ln_rstar)) ** q) + c * lnD

    (_, _, _, dose_c), cov = curve_fit(native, None, amp, p0=[-2.5, np.log(2.1), 1.9, 0.23], maxfev=60000)
    dose_c_err = float(np.sqrt(np.diag(cov))[3])
    span = float((dose.max() / dose.min()) ** dose_c)

    # the covariate is not purely separable: a mass x ln(dose) term lets the mass exponent slide with dose
    def interact(_, lnC, a, b, ln_rstar, q, c):
        return lnC + a * lnM + b * (lnM * lnDc) + 2 * lnR - np.log(1 + (rmsd / np.exp(ln_rstar)) ** q) + c * lnD

    theta, icov = curve_fit(interact, None, amp, p0=[-2, 1.0, 0.0, np.log(2.1), 1.9, 0.2], maxfev=200000)
    ierr = np.sqrt(np.diag(icov))
    interaction = {
        "mass_exp_at_ref": float(theta[1]),
        "coef": float(theta[2]),
        "coef_err": float(ierr[2]),
        "z": float(theta[2] / ierr[2]),
        "mass_exp_low": float(theta[1] + theta[2] * np.log(dose.min() / REF_DOSE)),
        "mass_exp_high": float(theta[1] + theta[2] * np.log(dose.max() / REF_DOSE)),
    }

    print(f"{len(rows)} converged pairs carry a mass; dose {dose.min():.0f}-{dose.max():.0f} e/A^2")
    print(f"dose covariate c = {dose_c:.3f} +- {dose_c_err:.3f}   d'^2 span across the range: {span:.2f}x")
    print(
        f"mass x dose interaction: coef {interaction['coef']:+.3f} (z {interaction['z']:+.1f}); "
        f"mass exponent {interaction['mass_exp_low']:.2f} (dose {dose.min():.0f}) -> "
        f"{interaction['mass_exp_at_ref']:.2f} (40) -> {interaction['mass_exp_high']:.2f} ({dose.max():.0f})"
    )

    write_result(
        args.out,
        {
            "dose_exponent": float(dose_c),
            "dose_exponent_err": dose_c_err,
            "reference_dose": REF_DOSE,
            "dose_min": float(dose.min()),
            "dose_max": float(dose.max()),
            "span_across_range": span,
            "interaction": interaction,
            "n_pairs": len(rows),
            "n_attempted": n_attempted,
            "n_converged": n_converged,
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
