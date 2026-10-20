"""Whether any structural predictor beats equation (1)'s Ca RMSD and mass.
    .venv/bin/python reproduction/experiments/panel/feature_regression.py --out results/feature_regression.json
`pair_features.py` computes many more structural numbers than equation (1) uses; this asks whether any
of them predicts `d^2` better. Scored by CROSS-VALIDATED `R^2` (ten folds, fixed seed), since in-sample
`R^2` only rises as features are added; the in-sample coefficient and error are still reported, as "is
this term real" and "does it improve prediction" differ. The residual factor (geometric SD of measured
over predicted) is what the manuscript quotes. `coherence` survives (cross terms cancel for a rigid
displacement, add for an incoherent one); substituting the moving mass for the total is REFUTED, since
`frac_moved` adds information rather than replacing mass. Rationale: docs/rationale/targets.md
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from reproduction.record import read_payload, write_result

_FEATURES = Path("results/pair_features.json")
_TARGET = "d2_marginal"
_STEPS = 6

DRIVER = "reproduction/experiments/panel/feature_regression.py"

LOGGED = frozenset(
    {
        "mass_kda",
        "ca_rmsd",
        "radius",
        "n_atoms",
        "rg",
        "dmax",
        "rmsd_all",
        "max_disp",
        "median_disp",
        "sum_sq_disp",
        "mass_weighted_disp",
        "d_rg",
        "com_shift",
        "moving_mass_1a",
        "moving_mass_2a",
    }
)
"""Features taken in logs, because the relation being fitted is a power law in them.

The rest are ratios already confined to a bounded range -- fractions, correlation coefficients,
shape scalars -- where a log would stretch the end nearest zero without making anything linear."""

CANDIDATES = (
    "mass_kda",
    "ca_rmsd",
    "radius",
    "n_atoms",
    "rg",
    "dmax",
    "asphericity",
    "acylindricity",
    "anisotropy",
    "rmsd_all",
    "max_disp",
    "median_disp",
    "sum_sq_disp",
    "mass_weighted_disp",
    "frac_moved_1a",
    "frac_moved_2a",
    "d_rg",
    "d_asphericity",
    "com_shift",
    "coherence",
    "participation_ratio",
    "translation_fraction",
    "moving_mass_1a",
    "moving_mass_2a",
)

N_FOLDS = 10
SEED = 0


def design(rows: list[dict], names: Sequence[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """The design matrix and target, restricted to rows finite on everything asked for.

    Args:
        rows: Feature rows from `pair_features.py`, with the derived columns already attached.
        names: Features to include.
        target: The column to predict, taken in logs.

    Returns:
        `(X, y)`.
    """

    def column(name: str) -> np.ndarray:
        values = np.array([row.get(name, np.nan) for row in rows], dtype=float)
        return np.log(np.clip(values, 1e-9, None)) if name in LOGGED else values

    y = np.log(np.array([row.get(target, np.nan) for row in rows], dtype=float))
    X = np.column_stack([column(name) for name in names]) if names else np.empty((len(rows), 0))
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1) if names else np.isfinite(y)
    return X[keep], y[keep]


def assess(X: np.ndarray, y: np.ndarray) -> dict:
    """Cross-validated fit quality, plus the in-sample coefficients.

    Args:
        X: Design matrix, without the intercept column.
        y: Target.

    Returns:
        `cv_r2`, `residual_factor`, `coefficients`, `errors`, `n_rows`.
    """
    folds = np.random.default_rng(SEED).permutation(len(y)) % N_FOLDS
    predicted = np.empty_like(y)
    for fold in range(N_FOLDS):
        test, train = folds == fold, folds != fold
        A = np.column_stack([np.ones(train.sum()), X[train]])
        beta, *_ = np.linalg.lstsq(A, y[train], rcond=None)
        predicted[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta

    residual_ss = float(((y - predicted) ** 2).sum())
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    # pseudo-inverse, because several candidates are exactly collinear so a plain inverse raises
    # `Singular matrix`; `rank` is reported so a caller sees a column that added no information
    cov = (float(resid @ resid) / dof) * np.linalg.pinv(A.T @ A)
    return {
        "rank": int(np.linalg.matrix_rank(A)),
        "n_columns": int(A.shape[1]),
        "cv_r2": 1.0 - residual_ss / float(((y - y.mean()) ** 2).sum()),
        "residual_factor": float(np.exp(np.sqrt(residual_ss / len(y)))),
        "coefficients": [float(v) for v in beta[1:]],
        "errors": [float(v) for v in np.sqrt(np.diag(cov))[1:]],
        "n_rows": len(y),
    }


def main(argv: Iterable[str] | None = None) -> int:
    """Rank the features, run forward selection, and compare the candidate models."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features",
        type=Path,
        default=_FEATURES,
        help="the pair_features.py record to regress, so a second corpus can be fitted without overwriting the first",
    )
    parser.add_argument("--out", type=Path, default=Path("results/feature_regression.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = read_payload(args.features)["rows"]
    rows = [r for r in rows if isinstance(r.get(_TARGET), (int, float)) and r[_TARGET] > 0]
    for row in rows:  # the derived columns the models need
        for cut in ("1a", "2a"):
            frac, mass = row.get(f"frac_moved_{cut}"), row.get("mass_kda")
            row[f"moving_mass_{cut}"] = (frac * mass) if (frac is not None and mass is not None) else None

    singles = {}
    for name in CANDIDATES:
        X, y = design(rows, [name], _TARGET)
        if len(y) > 10:
            singles[name] = {"r": float(np.corrcoef(X[:, 0], y)[0, 1]), "n_rows": len(y)}
    print(f"{len(rows)} rows. single-feature correlation with log {_TARGET}:")
    for name, row in sorted(singles.items(), key=lambda kv: -abs(kv[1]["r"]))[:8]:
        print(f"  {name:22s} r {row['r']:+.3f}   R^2 {row['r'] ** 2:.3f}")

    chosen: list[str] = []
    remaining = list(CANDIDATES)
    ladder = []
    for _ in range(_STEPS):
        scored = []
        for candidate in remaining:
            step = assess(*design(rows, chosen + [candidate], _TARGET))
            # a candidate that does not raise the rank is a linear combination of what is already
            # chosen; it cannot improve the fit and its coefficient would not be identified
            if step["rank"] == step["n_columns"]:
                scored.append((step["cv_r2"], candidate))
        if not scored:
            print("  (no candidate adds rank; stopping)")
            break
        best_r2, best = max(scored)
        chosen.append(best)
        remaining.remove(best)
        step = assess(*design(rows, chosen, _TARGET))
        ladder.append({"added": best, "features": list(chosen), **step})
        print(f"  + {best:22s} CV R^2 {step['cv_r2']:.4f}  residual {step['residual_factor']:.3f}")
        del best_r2

    MODELS = {
        "equation_1": ["ca_rmsd", "mass_kda"],
        "equation_1_plus_frac_moved": ["ca_rmsd", "mass_kda", "frac_moved_1a"],
        "equation_1_plus_coherence": ["ca_rmsd", "mass_kda", "coherence"],
        "equation_1_plus_both": ["ca_rmsd", "mass_kda", "frac_moved_1a", "coherence"],
        # predicted and REFUTED: substituting the moving mass for the total loses
        "moving_mass_replaces_mass": ["ca_rmsd", "moving_mass_1a"],
        "best_four": ["sum_sq_disp", "mass_kda", "frac_moved_1a", "coherence"],
    }
    models = {name: {"features": names, **assess(*design(rows, names, _TARGET))} for name, names in MODELS.items()}
    print("\nmodel comparisons:")
    for name, row in models.items():
        added = "  ".join(
            f"{nm} {c:+.3f}+-{e:.3f} ({abs(c / e):.1f}s)"
            for nm, c, e in zip(row["features"], row["coefficients"], row["errors"])
            if nm not in ("ca_rmsd", "mass_kda", "sum_sq_disp")
        )
        print(
            f"  {name:28s} CV R^2 {row['cv_r2']:.4f}  residual {row['residual_factor']:.3f}  n {row['n_rows']}  {added}"
        )

    # does a better fit change a decision? The relation is used to rank candidate targets, so what
    # matters is how often two models disagree on which target is easier, and who is right when they do
    ranking = {}
    for name in ("equation_1", "equation_1_plus_both", "best_four"):
        X, target = design(rows, MODELS[name], _TARGET)
        folds = np.random.default_rng(SEED).permutation(len(target)) % N_FOLDS
        out_of_fold = np.empty_like(target)
        for fold in range(N_FOLDS):
            test, train = folds == fold, folds != fold
            A = np.column_stack([np.ones(train.sum()), X[train]])
            beta, *_ = np.linalg.lstsq(A, target[train], rcond=None)
            out_of_fold[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta
        order = np.argsort(np.argsort(out_of_fold))
        truth_order = np.argsort(np.argsort(target))
        rho = float(np.corrcoef(order, truth_order)[0, 1])
        ranking[name] = {"spearman": rho, "predictions": [float(v) for v in out_of_fold]}
        print(f"  ranking: {name:24s} Spearman against truth {rho:+.4f}")

    # ONE row set for both models: each `design` drops rows non-finite on its OWN features, so two
    # differently-filtered arrays indexed together would compare different pairs of targets
    union = list(dict.fromkeys(MODELS["equation_1"] + MODELS["equation_1_plus_both"]))
    aligned, truth = design(rows, union, _TARGET)
    if len(truth) != len(ranking["equation_1"]["predictions"]):  # type: ignore[arg-type]
        print(
            f"  ranking comparison runs on {len(truth)} rows, against {len(rows)} featured: "
            "a model's own feature set drops more"
        )

    def _out_of_fold(names):
        columns = [union.index(n) for n in names]
        folds = np.random.default_rng(SEED).permutation(len(truth)) % N_FOLDS
        out = np.empty_like(truth)
        for fold in range(N_FOLDS):
            test, train = folds == fold, folds != fold
            A = np.column_stack([np.ones(train.sum()), aligned[train][:, columns]])
            beta, *_ = np.linalg.lstsq(A, truth[train], rcond=None)
            out[test] = np.column_stack([np.ones(test.sum()), aligned[test][:, columns]]) @ beta
        return out

    left = _out_of_fold(MODELS["equation_1"])
    right = _out_of_fold(MODELS["equation_1_plus_both"])
    pick = np.random.default_rng(3)
    i, j = pick.integers(0, len(truth), 200_000), pick.integers(0, len(truth), 200_000)
    i, j = i[i != j], j[i != j]
    correct = truth[i] > truth[j]
    disagree = (left[i] > left[j]) != (right[i] > right[j])
    augmented_right = ((right[i] > right[j]) == correct) & disagree
    accuracy = (float(((left[i] > left[j]) == correct).mean()), float(((right[i] > right[j]) == correct).mean()))
    ranking["pairwise"] = {
        "disagreement_fraction": float(disagree.mean()),
        "augmented_right_when_disagreeing": float(augmented_right.sum() / max(disagree.sum(), 1)),
        "accuracy_equation_1": accuracy[0],
        "accuracy_augmented": accuracy[1],
    }
    print(
        f"  the two disagree on {100 * disagree.mean():.1f}% of target pairs, and the augmented model is "
        f"right on {100 * augmented_right.sum() / max(disagree.sum(), 1):.0f}% of those"
    )
    print(f"  pairwise ranking accuracy {100 * accuracy[0]:.1f}% -> {100 * accuracy[1]:.1f}%")
    for row in ranking.values():
        row.pop("predictions", None)

    write_result(
        args.out,
        {
            "singles": singles,
            "forward_selection": ladder,
            "models": models,
            "ranking": ranking,
            "n_rows": len(rows),
            "n_folds": N_FOLDS,
            "seed": SEED,
        },
        driver=DRIVER,
        arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
