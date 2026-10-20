"""reproduction/experiments/reproduce.py: the runner builds each experiment's command at ONE operating point.

Command-building only -- never --run, so nothing is launched. What is checked is that a driver receives
the shared operating point (or its declared departure) and the right launcher for its target.
"""

from dataclasses import replace

import pytest
from conftest import load_driver

from reproduction.operating_point import SHARED


def test_to_flags_emits_the_operating_point_and_only_the_requested_knobs():
    """The full canonical set in order, and the subset a driver without --contrast asks for."""
    assert SHARED.to_flags() == ["--dose", "40", "--pixel-size", "2", "--rmsf", "0.5", "--contrast", "1"]
    assert SHARED.to_flags(("dose", "pixel_size", "rmsf")) == ["--dose", "40", "--pixel-size", "2", "--rmsf", "0.5"]


def test_a_departure_overrides_one_knob_and_leaves_the_rest_at_the_operating_point():
    """A registry departure is `replace(SHARED, ...)`: it moves one knob and nothing else."""
    runner = load_driver("reproduce.py")
    exp = runner.Experiment(
        name="real", driver="realdata/score_pool.py", op_knobs=("dose", "pixel_size"),
        settings=replace(SHARED, pixel_size=1.30),
    )  # fmt: skip
    argv = exp.argv()
    assert argv[argv.index("--pixel-size") + 1] == "1.3"
    assert argv[argv.index("--dose") + 1] == "40"  # the departure left dose at the operating point


def test_command_routes_each_target_to_its_launcher():
    """local runs the driver directly; orc/ribo delegate to the deploy launchers with the operating point."""
    runner = load_driver("reproduce.py")
    local = runner.Experiment(name="l", driver="panel/ip_converge.py", target="local", inputs=("--panel", "p.json"))
    assert runner.command(local)[:2] == [".venv/bin/python", "reproduction/experiments/panel/ip_converge.py"]
    assert runner.command(replace(local, target="orc"))[:2] == ["deploy/orc/submit.sh", "panel/ip_converge.py"]
    assert runner.command(replace(local, target="ribo"))[:2] == ["deploy/ribo/run.sh", "panel/ip_converge.py"]
    orc = runner.command(replace(local, target="orc"))
    assert "--panel" in orc and "p.json" in orc and "--pixel-size" in orc  # inputs and op point both ride along


def test_an_unknown_target_is_refused():
    runner = load_driver("reproduce.py")
    with pytest.raises(ValueError, match="unknown target"):
        runner.command(runner.Experiment(name="x", driver="d.py", target="cloud"))


def test_dry_run_is_the_default_and_prints_the_command_without_launching(capsys):
    """No --run, so main only prints -- the operating point and the launcher both appear in the line."""
    runner = load_driver("reproduce.py")
    assert runner.main(["--only", "ip_converge"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "deploy/orc/submit.sh panel/ip_converge.py" in out
    assert "--pixel-size 2" in out


def test_only_refuses_an_unknown_experiment_name():
    runner = load_driver("reproduce.py")
    with pytest.raises(SystemExit):
        runner.main(["--only", "does_not_exist"])
