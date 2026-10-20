"""survey_pool drains the survey per pair, writing one result file per pair and skipping done ones."""

import json
import sys
from types import SimpleNamespace

from conftest import load_driver

DRIVER = load_driver("panel/survey_pool.py")


def test_score_one_writes_a_diagnostic_stub_when_score_survey_produces_nothing(tmp_path, monkeypatch):
    out = tmp_path / "pair.json"
    # score_survey exits non-zero and writes no result (a dropped pair, or a failure)
    monkeypatch.setattr(
        DRIVER.subprocess, "run", lambda cmd, **k: SimpleNamespace(returncode=2, stderr="unrecognized arguments")
    )

    DRIVER.score_one("Dropped pair (1X/2Y)", out, SimpleNamespace(panel="p", data_dir="d", seed=0))

    rec = json.loads(out.read_text())
    assert rec["dropped"] == "Dropped pair (1X/2Y)"  # marked so the pool does not retry it forever
    assert rec["result"]["rows"] == []
    assert rec["returncode"] == 2 and "unrecognized" in rec["stderr_tail"]  # failure stays visible


def test_the_driver_scores_each_pair_once_and_skips_finished(tmp_path, monkeypatch):
    cands = [SimpleNamespace(name="A (1/2)"), SimpleNamespace(name="B (3/4)"), SimpleNamespace(name="C (5/6)")]
    monkeypatch.setattr(DRIVER, "load_panel", lambda *a, **k: (cands, []))

    scored = []

    def fake_score_one(name, out, args):  # stand in for the score_survey subprocess
        scored.append(name)
        out.write_text(json.dumps({"result": {"rows": [{"name": name}]}}))

    monkeypatch.setattr(DRIVER, "score_one", fake_score_one)

    out_dir = tmp_path / "survey_pool"
    argv = ["survey_pool", "--panel", str(tmp_path / "m.json"), "--out-dir", str(out_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    DRIVER.main()
    assert sorted(scored) == ["A (1/2)", "B (3/4)", "C (5/6)"]  # all three scored

    scored.clear()
    monkeypatch.setattr(sys, "argv", argv)
    DRIVER.main()
    assert scored == []  # second run skips every finished pair
