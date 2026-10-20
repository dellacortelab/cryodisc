"""Fit the mechanistic discriminability law to the convergent K_c survey.
    .venv/bin/python reproduction/experiments/panel/eq1_mechanistic.py --out results/eq1_mechanistic.json
Each converged pair's plateau is the crossover amplitude of its own d'^2(K) ladder, marginalized
over pose. Mass and rmsd enter as theory fixes them -- moving mass (matched filter) times squared
displacement (small-motion linear response) -- with a resolution-scale saturation. The survey is
collected at the single reference dose 40; the dose covariate was fit once on an archived multidose
design and found weak (reproduction/experiments/panel/dose_covariate.py), so no dose term is fit here. The frozen
form (mass^1, rmsd^2, r*=2, q=2) leaves one fitted number; the free fit is recorded as its validation.
The survey scored partial correspondences, so every pair is admitted only if its two models share
`MIN_OVERLAP` of their atoms: a d'^2 over part of a molecule is not comparable to one over all of it.

Rationale: docs/rationale/reproduction/experiments/survey_law.md
"""

import argparse

import numpy as np
from scipy.optimize import curve_fit

from reproduction.analysis.kc_survey import MIN_OVERLAP, REF_DOSE, census, gather, geometry, load_survey, plateau
from reproduction.record import write_result

DRIVER = "reproduction/experiments/panel/eq1_mechanistic.py"


def converged_deflections(records, admitted):
    """Plateau d'^2 and dose for every admitted converged pair with a fittable ladder: the survey's
    discriminability distribution, what the counting statistics span."""
    out = []
    for record in records:
        if record.get("status") != "converged" or record.get("name") not in admitted:
            continue
        name, dose = record.get("name"), record.get("dose")
        if not name or not dose:
            continue
        fitted = plateau(record.get("curve") or {})
        if fitted is None:
            continue
        out.append({"name": name, "d2_plateau": float(fitted[0]), "dose": dose})
    return out


def frozen_fit(rows):
    """The one-constant form on `rows` in closed form: least squares on ln C alone is the mean residual."""
    rmsd = np.array([r["ca_rmsd"] for r in rows])
    shape = np.log([r["mass_kda"] for r in rows]) + 2 * np.log(rmsd) - np.log(1 + (rmsd / 2.0) ** 2)
    target = np.log([r["d2_plateau"] for r in rows])
    ln_c = float(np.mean(target - shape))
    return {
        "n": len(rows),
        "C": float(np.exp(ln_c)),
        "r2": r2_of(target, shape + ln_c),
        "residual_factor": float(np.exp(np.std(target - shape - ln_c))),
    }


def r2_of(target, prediction):
    """Coefficient of determination in the space the fit works in (log d'^2)."""
    return float(1 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def main() -> int:
    """Fit the ladder of forms free -> frozen, print them, and record the fit with its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", default="results/kc_survey/*.json")
    parser.add_argument("--pool", default="results/survey_pool/*.json")
    parser.add_argument("--panels", nargs="+", default=["data/survey_*.json"])
    parser.add_argument("--out", default="results/eq1_mechanistic.json")
    args = parser.parse_args()

    records = load_survey(args.survey)
    n_attempted, n_converged = census(records)
    converged = sorted({r["name"] for r in records if r.get("status") == "converged" and r.get("name")})
    shapes = geometry(converged, args.pool, args.panels)
    admitted = {name: shape for name, shape in shapes.items() if shape[1] >= MIN_OVERLAP}
    rows = gather(records, admitted)
    all_deflections = converged_deflections(records, admitted)
    mass = np.array([r["mass_kda"] for r in rows])
    rmsd = np.array([r["ca_rmsd"] for r in rows])
    amp = np.log([r["d2_plateau"] for r in rows])
    lnM, lnR = np.log(mass), np.log(rmsd)
    n = len(rows)

    # every survey pair is collected at the single reference dose, so the amplitude needs no rescaling;
    # the dose covariate was fit once on the archived multidose design (reproduction/experiments/panel/dose_covariate.py)
    amp40 = amp

    def fit_ref(free_rstar, free_q, free_mass):
        # log d'^2 at dose 40 = lnC + a*lnM + 2*lnR - ln(1 + (rmsd/r*)^q)
        def model(_, *theta):
            i = 0
            lnC = theta[i]
            i += 1
            a = theta[i] if free_mass else 1.0
            i += free_mass
            r_star = np.exp(theta[i]) if free_rstar else 2.0
            i += free_rstar
            qq = theta[i] if free_q else 2.0
            return lnC + a * lnM + 2 * lnR - np.log(1 + (rmsd / r_star) ** qq)

        p0 = [amp40.mean() - lnM.mean()]
        if free_mass:
            p0.append(1.0)
        if free_rstar:
            p0.append(np.log(2.1))
        if free_q:
            p0.append(1.9)
        theta, cov = curve_fit(model, None, amp40, p0=p0, maxfev=60000)
        pred = model(None, *theta)
        err = np.sqrt(np.diag(cov))
        out = {
            "n_params": len(theta),
            "r2": r2_of(amp40, pred),
            "residual_factor": float(np.exp(np.std(amp40 - pred))),
            "C": float(np.exp(theta[0])),
            "C_err": float(err[0] * np.exp(theta[0])),
        }
        j = 1
        if free_mass:
            out["mass_exp"] = float(theta[j])
            out["mass_exp_err"] = float(err[j])
            j += 1
        else:
            out["mass_exp"] = 1.0
        if free_rstar:
            out["r_star"] = float(np.exp(theta[j]))
            out["r_star_err"] = float(np.exp(theta[j]) * err[j])
            j += 1
        else:
            out["r_star"] = 2.0
        if free_q:
            out["q"] = float(theta[j])
            out["q_err"] = float(err[j])
        else:
            out["q"] = 2.0
        return out

    canonical = fit_ref(free_rstar=False, free_q=False, free_mass=False)
    # the floor's sensitivity: the same frozen form on every converged pair, overlap unchecked
    if not np.isclose(frozen_fit(rows)["C"], canonical["C"], rtol=1e-6):
        raise SystemExit("frozen_fit no longer reproduces the canonical fit; the sensitivity would not compare")
    unfloored = frozen_fit(gather(records, shapes))
    free_shape = fit_ref(free_rstar=True, free_q=True, free_mass=False)
    free_mass = fit_ref(free_rstar=True, free_q=True, free_mass=True)

    # the product power law, for the comparison the Results draws against
    design = np.column_stack([np.ones(n), lnM + lnR])
    (pl_c, pl_e), *_ = np.linalg.lstsq(design, amp40, rcond=None)
    power_law = {
        "intercept": float(pl_c),
        "product_exp": float(pl_e),
        "r2": r2_of(amp40, design @ [pl_c, pl_e]),
        "residual_factor": float(np.exp(np.std(amp40 - design @ [pl_c, pl_e]))),
    }

    # two independent exponents: decoupling mass and RMSD still cannot bend, so it does not help
    design2 = np.column_stack([np.ones(n), lnM, lnR])
    (pl2_c, pl2_m, pl2_r), *_ = np.linalg.lstsq(design2, amp40, rcond=None)
    power_law_two = {
        "intercept": float(pl2_c),
        "mass_exp": float(pl2_m),
        "rmsd_exp": float(pl2_r),
        "r2": r2_of(amp40, design2 @ [pl2_c, pl2_m, pl2_r]),
        "residual_factor": float(np.exp(np.std(amp40 - design2 @ [pl2_c, pl2_m, pl2_r]))),
    }

    # single predictors: Ca RMSD alone is the field's convention, mass alone the mass-only strawman
    def single(column):
        mat = np.column_stack([np.ones(n), column])
        coef, *_ = np.linalg.lstsq(mat, amp40, rcond=None)
        return {
            "intercept": float(coef[0]),
            "exp": float(coef[1]),
            "r2": r2_of(amp40, mat @ coef),
            "residual_factor": float(np.exp(np.std(amp40 - mat @ coef))),
        }

    singles = {"ca_rmsd": single(lnR), "mass_kda": single(lnM)}

    ratio = amp40 - (np.log(canonical["C"]) + lnM + 2 * lnR - np.log(1 + (rmsd / 2.0) ** 2))
    print(
        f"survey funnel: {n_attempted} pairs attempted, {n_converged} converged, {len(admitted)} share "
        f">= {MIN_OVERLAP:.0%} of their atoms, {n} enter the fit"
    )
    print(f"{n} converged pairs, plateau = crossover amplitude at reference dose {REF_DOSE:g}")
    for name, block in (
        ("canonical (1 param)", canonical),
        ("free shape (3)", free_shape),
        ("free mass too (4)", free_mass),
    ):
        extra = f"mass^{block['mass_exp']:.3f} r*={block['r_star']:.2f} q={block['q']:.2f}"
        print(f"  {name:20s} C={block['C']:.4f}  {extra}  R^2={block['r2']:.4f}  x/{block['residual_factor']:.2f}")
    print(
        f"  product power law     exp={power_law['product_exp']:.3f}  R^2={power_law['r2']:.4f}  x/{power_law['residual_factor']:.2f}"
    )
    print(
        f"  two free exponents    mass^{power_law_two['mass_exp']:.3f} rmsd^{power_law_two['rmsd_exp']:.3f}  R^2={power_law_two['r2']:.4f}  x/{power_law_two['residual_factor']:.2f}"
    )
    for name, block in singles.items():
        print(f"  {name} alone         exp={block['exp']:.3f}  R^2={block['r2']:.4f}  x/{block['residual_factor']:.2f}")
    print(
        f"  frozen residual spans {np.exp(ratio.min()):.2f}x..{np.exp(ratio.max()):.2f}x, median x/{np.exp(np.median(np.abs(ratio))):.2f}"
    )
    print(
        f"  no overlap floor: {unfloored['n']} pairs, C={unfloored['C']:.4f}  R^2={unfloored['r2']:.4f}  "
        f"x/{unfloored['residual_factor']:.2f}"
    )
    print(
        f"  mass range {mass.min():.1f}-{mass.max():.1f} kDa (median {np.median(mass):.1f}), rmsd {rmsd.min():.2f}-{rmsd.max():.2f} A"
    )

    write_result(
        args.out,
        {
            "n_rows": n,
            "n_attempted": n_attempted,
            "n_converged": n_converged,
            "min_overlap": MIN_OVERLAP,
            "n_admitted": len(admitted),
            "unfloored": unfloored,
            "reference_dose": REF_DOSE,
            "canonical": canonical,
            "free_shape": free_shape,
            "free_mass": free_mass,
            "power_law": power_law,
            "power_law_two": power_law_two,
            "singles": singles,
            "rows": rows,
            "converged_deflections": all_deflections,
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
