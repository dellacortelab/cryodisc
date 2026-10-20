"""Run every experiment at one operating point, instead of replaying each one's stored args.

Supersedes the replay-from-records approach (deploy/orc/build_replay_manifest.py): rather than
reconstructing each invocation from the operating point baked into its result JSON -- which is
divergent, since each driver stored its own pixel/dose/nodes -- this declares the operating point
ONCE (`SHARED`) and launches every registered experiment with it. Each experiment runs as a
subprocess, never an import (reproduction/experiments/ forbids sibling imports), so this stays compliant with
tests/test_experiments_layout.py. Rationale: docs/planning/unified_reproduction.md.
"""

import argparse
import shlex
import subprocess
from dataclasses import dataclass, replace

# the one operating point, shared with every driver's argparse defaults (reproduction/operating_point.py):
# declared once so a knob cannot drift between a driver's default and what the runner emits
from reproduction.operating_point import OP_FLAGS, SHARED, RunSettings


@dataclass(frozen=True)
class Experiment:
    """One registered experiment: a driver, its inputs, and the operating point it runs at.

    Args:
        name: Registry key, used by `--only`.
        driver: Path under reproduction/experiments/, e.g. "panel/ip_converge.py".
        inputs: Per-experiment argv beyond the operating point (panels, out paths, ...).
        op_knobs: Which `SHARED` knobs this driver accepts as flags.
        settings: The operating point; `SHARED`, or a named departure via `replace(SHARED, ...)`.
        target: Where to launch: "local", "orc", or "ribo".
        note: A human note -- a declared departure, an open question, or a migration flag.
    """

    name: str
    driver: str
    inputs: tuple[str, ...] = ()
    op_knobs: tuple[str, ...] = OP_FLAGS
    settings: RunSettings = SHARED
    target: str = "orc"
    note: str = ""

    def argv(self) -> list[str]:
        """The driver's full argument list: operating point first, then per-experiment inputs."""
        return [*self.settings.to_flags(self.op_knobs), *self.inputs]


_LAUNCHER = {"orc": ["deploy/orc/submit.sh"], "ribo": ["deploy/ribo/run.sh"]}


def command(exp: Experiment) -> list[str]:
    """The shell command that launches `exp` on its target.

    Local runs invoke the driver directly; cluster runs delegate to the deploy launchers, which take
    the driver path relative to reproduction/experiments/ and pass everything after it untouched.
    """
    if exp.target == "local":
        return [".venv/bin/python", f"reproduction/experiments/{exp.driver}", *exp.argv()]
    if exp.target in _LAUNCHER:
        return [*_LAUNCHER[exp.target], exp.driver, *exp.argv()]
    raise ValueError(f"{exp.name}: unknown target {exp.target!r}; expected local, orc or ribo")


# The registry: every physical-scorer experiment, at the shared operating point unless it declares a
# departure. Inputs are reconstructed from the CURRENT tree (data/, build.py, the figure drivers'
# inputs), not the stale deploy/. Real-data --meta/--root are CLUSTER-ONLY -- the particle stacks are
# not in the repo -- and marked with CLUSTER; fill them in before a real launch. Reducers, figure
# drivers and preprocess are not rows here.
CLUSTER = "CLUSTER"  # a path only on the compute cluster (the particle stacks); set before launching

REGISTRY: list[Experiment] = [
    # --- the survey population (Figure 5), over data/survey_canonical.json ---
    Experiment(
        name="kc_survey",
        driver="panel/kc_survey.py",
        inputs=(
            "--panels",
            "data/survey_canonical.json",
            "--out-dir",
            "results/kc_survey",
            "--rise",
            "0.025",
            "--min-stop-bank",
            "100000",
            "--cap-bank",
            "2000000",
        ),
        op_knobs=("dose", "rmsf"),  # now runs at the SHARED dose (40); contrast fixed at 1 inside the driver
        target="orc",
        note="Figure 5 K_c survey over the canonical population at the shared dose (40); contrast fixed "
        "at 1, grid chased to a measured plateau (rise 0.025, 100k floor, 2M cap).",
    ),
    Experiment(
        name="survey_pool",
        driver="panel/survey_pool.py",
        inputs=("--panel", "data/survey_canonical.json", "--out-dir", "results/survey_pool"),
        op_knobs=(),  # the operating point is inside survey_pool.SCORE_ARGS, which now reads SHARED
        target="orc",
        note="target-ranking survey; wraps score_survey per pair. Operating point in survey_pool.SCORE_ARGS.",
    ),
    # --- simulated benchmarks ---
    Experiment(
        name="ip_converge",
        driver="panel/ip_converge.py",
        inputs=("--panel", "data/survey_canonical.json", "--out", "results/ip_converge.json"),
        target="orc",
        note="per-bank marginal scorer (also kc_survey's primitive); converges over --banks.",
    ),
    Experiment(
        name="difference_spectrum",
        driver="benchmark/difference_spectrum.py",
        inputs=("--panels", "data/survey_canonical.json", "--out", "results/difference_spectrum.json"),
        op_knobs=("dose", "pixel_size", "rmsf"),  # no --contrast flag
        target="orc",
        note="structures-only difference spectrum over the population; feeds panel/grid_difficulty.py.",
    ),
    # --- real data: cas9, full-map route + pose MARGINALIZATION (EMPIAR-11779/11780) ---
    # The headline is marginal, not known-pose. Full maps are built from each state's two half maps,
    # then both pools are scored over a fixed 4M-node bank (the deepest converged rung the ladder was
    # read at). The known-pose oracle, the d'^2 upper bound, is deploy/orc/_cas9_fullmaps.sh -- a
    # deliberate comparison, not a headline record.
    Experiment(
        name="build_cas9_full_maps",
        driver="preprocess/build_full_maps.py",
        inputs=(
            "--state-a-halves",
            "data/maps/cas9/J9_005_volume_map_half_A.mrc",
            "data/maps/cas9/J9_005_volume_map_half_B.mrc",
            "--state-b-halves",
            "data/maps/cas9/J9_006_volume_map_half_A.mrc",
            "data/maps/cas9/J9_006_volume_map_half_B.mrc",
            "--target-pixel-size",
            "1.3",
            "--out-a",
            "results/cas9_fullmap_maps/cas9_a.mrc",
            "--out-b",
            "results/cas9_fullmap_maps/cas9_b.mrc",
            "--record",
            "results/cas9_fullmap_maps/maps.json",
        ),
        op_knobs=(),  # a map builder: no scoring operating point
        target="local",
        note="full maps from the half-map pairs (6nt=J9_005=state A, 12nt=J9_006=state B).",
    ),
    Experiment(
        name="score_pool_cas9_6nt",
        driver="realdata/score_pool.py",
        inputs=(
            "--map-a",
            "results/cas9_fullmap_maps/cas9_a.mrc",
            "--map-b",
            "results/cas9_fullmap_maps/cas9_b.mrc",
            "--meta",
            f"{CLUSTER}/cas9_6nt_splitB.cs",
            "--root",
            f"{CLUSTER}/CS-e9-spcas9-11779-2",
            "--calibrate",
            "poses",
            "--limit",
            "20000",
            "--assembly",
            "1",
            # the score's band limit is a DECLARED operating point (main.md: "to a stated band limit,
            # not Nyquist"): 10 A defocus grouping, 8 A band stop. Omitting these floods the score with
            # high-frequency template mismatch and halves the real separation.
            "--k-max",
            "0.1",
            "--band-limit",
            "0.125",
            "--shift-window",
            "4",
            "--shift-step",
            "1.0",
            "--fixed-bank",
            "4000000",
            "--n-shuffles",
            "0",
            "--n-iterations",
            "60",
            "--out",
            "results/rerun_cas9_hoRepro_6nt.json",
            "--log-ratio",
            "results/rerun_cas9_hoRepro_6nt.npy",
        ),
        op_knobs=("pixel_size",),  # real data keeps its native pixel; dose defaults to SHARED (40)
        settings=replace(SHARED, pixel_size=1.30),  # cas9 native pixel
        target="orc",
        note="EMPIAR-11779 (6nt), full-map route, pose-marginalized at 4M nodes. --meta/--root are CLUSTER-ONLY.",
    ),
    Experiment(
        name="score_pool_cas9_12nt",
        driver="realdata/score_pool.py",
        inputs=(
            "--map-a",
            "results/cas9_fullmap_maps/cas9_a.mrc",
            "--map-b",
            "results/cas9_fullmap_maps/cas9_b.mrc",
            "--meta",
            f"{CLUSTER}/cas9_12nt_splitB.cs",
            "--root",
            f"{CLUSTER}/CS-e9-spcas9-11780-2",
            "--calibrate",
            "poses",
            "--limit",
            "20000",
            "--assembly",
            "1",
            "--k-max",
            "0.1",
            "--band-limit",
            "0.125",
            "--shift-window",
            "4",
            "--shift-step",
            "1.0",  # declared 10 A band limit
            "--fixed-bank",
            "4000000",
            "--n-shuffles",
            "0",
            "--n-iterations",
            "60",
            "--out",
            "results/rerun_cas9_hoRepro_12nt.json",
            "--log-ratio",
            "results/rerun_cas9_hoRepro_12nt.npy",
        ),
        op_knobs=("pixel_size",),
        settings=replace(SHARED, pixel_size=1.30),
        target="orc",
        note="EMPIAR-11780 (12nt), same full maps as 6nt, pose-marginalized at 4M nodes. --meta/--root are CLUSTER-ONLY.",
    ),
    # --- recovery off the cas9 pools (precomputed Lambda; no bank to score) ---
    Experiment(
        name="known_fraction_cas9",
        driver="benchmark/known_fraction.py",
        inputs=(
            "--log-ratio-a",
            "results/rerun_cas9_hoRepro_6nt.npy",
            "--log-ratio-b",
            "results/rerun_cas9_hoRepro_12nt.npy",
            "--fractions",
            "0.0",
            "0.125",
            "0.25",
            "0.375",
            "0.5",
            "0.625",
            "0.75",
            "0.875",
            "1.0",
            "--out",
            "results/rerun_purity_cas9_hoRepro.json",
        ),
        op_knobs=(),  # read arm: reads precomputed Lambda, no operating point
        target="local",
        note="real-data purity off the cas9 pools; depends on score_pool_cas9_{6nt,12nt} running first.",
    ),
    Experiment(
        name="calculator_check",
        driver="benchmark/calculator_check.py",
        inputs=("--out", "results/calculator_check.json"),
        op_knobs=(),  # read arm: resamples the committed SpCas9 and recovery-system pools
        target="local",
        note="the particle calculator on real pools: two-pool estimate with held-out references, against its Fisher prediction.",
    ),
    # --- forward-model validation: the multislice gate cited in main.md (a prose claim, not a figure) ---
    # Gate renders at its OWN resolution (pixel 1.5, box 64), not the survey operating point: compare must
    # match the reference render's optics. The reference banks are committed, so compare runs local.
    Experiment(
        name="gate_8WJ1_df5000",
        driver="forward/gate.py",
        inputs=(
            "compare",
            "--structure",
            "data/pdb/8WJ1.pdb",
            "--reference",
            "banks/8WJ1_df5000_sl1.5.npz",
            "--box-size",
            "64",
            "--voltage",
            "300",
            "--defocus",
            "5000",
            "--nodes",
            "16",
            "--out",
            "results/gate_8WJ1_df5000.json",
        ),
        op_knobs=("pixel_size",),
        settings=replace(SHARED, pixel_size=1.5),
        target="local",
        note="two-engine forward-model gate cited in main.md (agreement to 8 A at 5000 A defocus). The "
        "multislice --reference is committed in banks/, so compare runs local.",
    ),
    Experiment(
        name="gate_8WJ1_df12000",
        driver="forward/gate.py",
        inputs=(
            "compare",
            "--structure",
            "data/pdb/8WJ1.pdb",
            "--reference",
            "banks/8WJ1_df12000_sl1.5.npz",
            "--box-size",
            "64",
            "--voltage",
            "300",
            "--defocus",
            "12000",
            "--nodes",
            "16",
            "--out",
            "results/gate_8WJ1_df12000.json",
        ),
        op_knobs=("pixel_size",),
        settings=replace(SHARED, pixel_size=1.5),
        target="local",
        note="the df12000 twin; certifies nothing past the CTF sampling limit, bounding the regime the "
        "templates are used in (the other half of the main.md claim).",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only these registry names")
    parser.add_argument(
        "--run", action="store_true", help="actually launch each experiment; the default prints the commands (dry run)"
    )
    args = parser.parse_args(argv)

    names = set(args.only) if args.only else None
    if names:
        known = {e.name for e in REGISTRY}
        unknown = names - known
        if unknown:
            parser.error(f"unknown experiment name(s): {sorted(unknown)}; known: {sorted(known)}")
    chosen = [e for e in REGISTRY if names is None or e.name in names]

    print(
        f"# {'launching' if args.run else 'dry run'}: {len(chosen)} experiment(s) at "
        f"dose={SHARED.dose:g} pixel={SHARED.pixel_size:g} rmsf={SHARED.rmsf:g} contrast={SHARED.contrast:g}"
    )
    for exp in chosen:
        if exp.note:
            print(f"# {exp.name}: {exp.note}")
        cmd = command(exp)
        print(" ".join(shlex.quote(c) for c in cmd))
        if args.run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
