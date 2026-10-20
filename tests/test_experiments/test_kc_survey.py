"""kc_survey's within-pair checkpoint: a pair killed mid-ladder resumes, it does not restart."""

import json
import sys
from types import SimpleNamespace

import pytest
from conftest import load_driver

DRIVER = load_driver("panel/kc_survey.py")


def _args(out_dir, cap_bank=64128):
    """A namespace with the fields measure_pair reads (rise 0.15 keeps the pre-flatness stop points)."""
    return SimpleNamespace(
        seed=0,
        dose=40.0,
        rise=0.15,
        min_stop_bank=0,
        cap_bank=cap_bank,
        n_particles=4000,
        rmsf=0.5,
        out_dir=out_dir,
        data_dir="data/pdb",
    )


def _scorer(calls, plateau=2000.0, fail_on=None):
    """A fake score_banks: d2(bank) = 100*bank/(bank+plateau), rising then flat.

    Records each requested bank list in `calls`; raises RuntimeError on the `fail_on`-th call to
    stand in for a job killed mid-pair.
    """

    def score_banks(panel, name, dose, banks, args, tmp):
        calls.append(list(banks))
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("killed mid-pair")
        return {
            "snr": 12.0,
            "rmsd": 3.0,
            "rungs": [{"k": b, "d2": 100.0 * b / (b + plateau)} for b in banks],
        }

    return score_banks


def test_an_uninterrupted_pair_converges_and_writes_a_terminal_record(tmp_path, monkeypatch):
    out = tmp_path / "pair.json"
    calls = []
    monkeypatch.setattr(DRIVER, "score_banks", _scorer(calls))

    rec = DRIVER.measure_pair("panel.json", "PAIR", _args(tmp_path), tmp_path / "_tmp.json", out)

    assert rec["status"] == "converged"
    assert list(rec["curve"]) == ["504", "2016", "8064", "33384", "64128"]
    assert json.loads(out.read_text())["status"] == "converged"


def test_a_pair_killed_mid_ladder_leaves_a_resumable_checkpoint(tmp_path, monkeypatch):
    out = tmp_path / "pair.json"
    calls = []
    monkeypatch.setattr(DRIVER, "score_banks", _scorer(calls, fail_on=3))  # dies scoring 64128

    with pytest.raises(RuntimeError):
        DRIVER.measure_pair("panel.json", "PAIR", _args(tmp_path), tmp_path / "_tmp.json", out)

    ckpt = json.loads(out.read_text())
    assert ckpt["status"] == "in_progress"
    assert list(ckpt["curve"]) == ["504", "2016", "8064", "33384"]  # 64128 never landed
    assert ckpt["step"] == 5  # resume position: next iteration scores ladder[4] = 64128


def test_resuming_continues_from_the_checkpoint_without_rescoring(tmp_path, monkeypatch):
    out = tmp_path / "pair.json"
    args, tmp = _args(tmp_path), tmp_path / "_tmp.json"

    killed = []
    monkeypatch.setattr(DRIVER, "score_banks", _scorer(killed, fail_on=3))
    with pytest.raises(RuntimeError):
        DRIVER.measure_pair("panel.json", "PAIR", args, tmp, out)

    resumed = []
    monkeypatch.setattr(DRIVER, "score_banks", _scorer(resumed))
    rec = DRIVER.measure_pair("panel.json", "PAIR", args, tmp, out)

    assert rec["status"] == "converged"
    assert resumed == [[64128]]  # only the unscored top rung, nothing re-scored
    assert list(rec["curve"]) == ["504", "2016", "8064", "33384", "64128"]

    # a resumed pair reads the identical curve an uninterrupted one would
    fresh_out = tmp_path / "fresh.json"
    monkeypatch.setattr(DRIVER, "score_banks", _scorer([]))
    fresh = DRIVER.measure_pair("panel.json", "PAIR", args, tmp, fresh_out)
    assert rec["curve"] == fresh["curve"]
    assert rec["dose"] == fresh["dose"] == 40.0  # fixed dose, consistent across resume and fresh runs


def test_a_pair_still_rising_at_the_cap_is_censored(tmp_path, monkeypatch):
    out = tmp_path / "pair.json"
    calls = []
    # a large plateau keeps d2 rising past the cap, so it never converges
    monkeypatch.setattr(DRIVER, "score_banks", _scorer(calls, plateau=5.0e6))

    rec = DRIVER.measure_pair("panel.json", "PAIR", _args(tmp_path, cap_bank=64128), tmp_path / "_tmp.json", out)

    assert rec["status"] == "censored"
    assert rec["kc"].startswith(">")


def _run_writing(tmp, returncode, rows, stderr=""):
    """A fake subprocess.run: write `rows` to ip_converge's out file and report `returncode`."""

    def fake_run(cmd, capture_output, text, check=False):
        if rows is not None:
            tmp.write_text(json.dumps({"rows": rows}))
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    return fake_run


def test_score_banks_returns_the_row_on_success(tmp_path, monkeypatch):
    tmp = tmp_path / "_tmp.json"
    row = {"snr": 12.0, "rmsd": 3.0, "rungs": [{"k": 504, "d2": 0.4}]}
    monkeypatch.setattr(DRIVER.subprocess, "run", _run_writing(tmp, 0, [row]))

    assert DRIVER.score_banks("panel.json", "PAIR", 40.0, [504], _args(tmp_path), tmp) == row


def test_score_banks_empty_rows_raises_scorecrash(tmp_path, monkeypatch):
    """ip_converge exiting 0 but writing no rows is transient (the name is always in the panel), so it
    must retry -- never be recorded as a terminal drop that discards the pair's partial curve."""
    tmp = tmp_path / "_tmp.json"
    monkeypatch.setattr(DRIVER.subprocess, "run", _run_writing(tmp, 0, []))

    with pytest.raises(DRIVER.ScoreCrash):
        DRIVER.score_banks("panel.json", "PAIR", 40.0, [504], _args(tmp_path), tmp)


def test_score_banks_nonzero_exit_raises_scorecrash(tmp_path, monkeypatch):
    tmp = tmp_path / "_tmp.json"
    monkeypatch.setattr(DRIVER.subprocess, "run", _run_writing(tmp, 1, None, stderr="CUDA OOM"))

    with pytest.raises(DRIVER.ScoreCrash):
        DRIVER.score_banks("panel.json", "PAIR", 40.0, [504], _args(tmp_path), tmp)


def test_a_crash_raises_and_leaves_no_terminal_record(tmp_path, monkeypatch):
    """A ScoreCrash (OOM/cold-compile/preempt) propagates and writes nothing, so the pair is retried."""
    out = tmp_path / "pair.json"

    def boom(*a, **k):
        raise DRIVER.ScoreCrash("cold XLA compile OOM")

    monkeypatch.setattr(DRIVER, "score_banks", boom)

    with pytest.raises(DRIVER.ScoreCrash):
        DRIVER.measure_pair("panel.json", "PAIR", _args(tmp_path), tmp_path / "_tmp.json", out)

    assert not out.exists()  # no terminal record left behind
    assert not DRIVER._done(out)  # so a requeue re-runs it rather than skipping


# --- the driver's queue loop: --shard partition and skip-done ---------------------------------


def _fake_measure(measured):
    """A measure_pair stub that writes a terminal record and logs the name it scored."""

    def fake_measure(panel, name, args, tmp, out):
        rec = {
            "name": name,
            "dose": 40.0,
            "snr": 5.0,
            "rmsd": 3.0,
            "kc": "504-2016",
            "status": "converged",
            "seconds": 0.1,
            "curve": {},
        }
        out.write_text(json.dumps(rec))
        measured.append(name)
        return rec

    return fake_measure


def test_the_driver_shards_and_skips_finished_pairs(tmp_path, monkeypatch):
    """--shard i/N partitions the sorted pairs across workers, and a pair whose terminal record is
    already on disk is skipped -- so a requeued shard resumes rather than rescoring."""
    panel = tmp_path / "panel.json"
    panel.write_text("{}")
    cands = [SimpleNamespace(name=n) for n in ("P1", "P2", "P3", "P4")]
    monkeypatch.setattr(DRIVER, "load_panel", lambda *a, **k: (cands, None))

    measured: list[str] = []
    monkeypatch.setattr(DRIVER, "measure_pair", _fake_measure(measured))
    out_dir = tmp_path / "out"

    def run_shard(shard):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kc_survey",
                "--panels",
                str(panel),
                "--out-dir",
                str(out_dir),
                "--shard",
                shard,
            ],
        )
        DRIVER.main()

    run_shard("0/2")  # sorted P1..P4 -> shard 0 takes every other one, P1 and P3
    assert measured == ["P1", "P3"]

    measured.clear()
    run_shard("0/2")  # both now have terminal records: nothing is rescored
    assert measured == []

    measured.clear()
    run_shard("1/2")  # the complementary shard takes P2 and P4
    assert measured == ["P2", "P4"]


def test_a_crashing_pair_does_not_take_the_worker_down(tmp_path, monkeypatch):
    """A ScoreCrash on one pair is caught in the worker loop: no terminal record, the shard continues."""
    panel = tmp_path / "panel.json"
    panel.write_text("{}")
    cands = [SimpleNamespace(name=n) for n in ("P1", "P2", "P3")]
    monkeypatch.setattr(DRIVER, "load_panel", lambda *a, **k: (cands, None))

    measured: list[str] = []

    def measure(panel, name, args, tmp, out):
        if name == "P2":
            raise DRIVER.ScoreCrash("boom on P2")
        rec = {
            "name": name,
            "dose": 40.0,
            "snr": 5.0,
            "rmsd": 3.0,
            "kc": "504-2016",
            "status": "converged",
            "seconds": 0.1,
            "curve": {},
        }
        out.write_text(json.dumps(rec))
        measured.append(name)
        return rec

    monkeypatch.setattr(DRIVER, "measure_pair", measure)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", ["kc_survey", "--panels", str(panel), "--out-dir", str(out_dir)])
    DRIVER.main()  # must not raise, despite P2 crashing

    assert measured == ["P1", "P3"]  # the worker scored the survivors and moved past the crash
    p2 = out_dir / f"{__import__('hashlib').sha1(b'P2').hexdigest()[:10]}.json"
    assert not p2.exists()  # P2 left no terminal record, so a later pass retries it
