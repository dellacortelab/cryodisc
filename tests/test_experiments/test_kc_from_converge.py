"""kc_from_converge reads both output shapes -- ip_converge mappings and kc_survey per-pair records."""

import json

from conftest import load_driver

DRIVER = load_driver("panel/kc_from_converge.py")


def _kc_survey_record(name, curve, status="converged"):
    return {
        "name": name,
        "status": status,
        "curve": {str(k): v for k, v in curve.items()},
        "dose": 40.0,
        "snr": 10.0,
        "rmsd": 3.0,
        "kc": "x",
        "step": 5,
    }


def _ip_converge_doc(name, rungs):
    return {"rows": [{"name": name, "rungs": [{"k": k, "d2": d2} for k, d2 in rungs]}]}


def test_load_curves_reads_a_kc_survey_survey_dir(tmp_path):
    rec = _kc_survey_record("Kinase (1ABC/2DEF)", {504: 10.0, 8064: 80.0, 64128: 96.0})
    (tmp_path / "abc123.json").write_text(json.dumps(rec))

    by_key, name_of = DRIVER.load_curves([tmp_path])

    assert by_key["Kinase (1ABC/2DEF)"] == {504: [10.0], 8064: [80.0], 64128: [96.0]}
    assert name_of["Kinase (1ABC/2DEF)"] == "Kinase (1ABC/2DEF)"


def test_load_curves_reads_an_ip_converge_mapping_dir(tmp_path):
    doc = _ip_converge_doc("Kinase (1ABC/2DEF)", [(504, 10.0), (8064, 80.0)])
    (tmp_path / "ip_converge_KIN_a1.0.json").write_text(json.dumps(doc))

    by_key, name_of = DRIVER.load_curves([tmp_path])

    assert by_key["KIN"] == {504: [10.0], 8064: [80.0]}
    assert name_of["KIN"] == "Kinase (1ABC/2DEF)"


def test_load_curves_skips_in_progress_records_and_scratch_files(tmp_path):
    (tmp_path / "done.json").write_text(json.dumps(_kc_survey_record("A (1/2)", {504: 5.0})))
    (tmp_path / "partial.json").write_text(json.dumps(_kc_survey_record("B (3/4)", {504: 5.0}, status="in_progress")))
    (tmp_path / "_tmp_ip_converge_999.json").write_text(json.dumps(_ip_converge_doc("C", [(504, 5.0)])))

    by_key, _ = DRIVER.load_curves([tmp_path])

    assert set(by_key) == {"A (1/2)"}  # in-progress and the leading-underscore scratch file skipped


def test_kc_for_pair_brackets_a_converged_curve(tmp_path):
    info = DRIVER.kc_for_pair({504: [10.0], 2016: [40.0], 8064: [80.0], 33384: [94.0], 64128: [96.5]})
    assert info["status"] == "converged"  # top-octave gain < RISE
    assert "-" in info["kc"] and not info["kc"].startswith(">")
    assert abs(info["retention_8064"] - 80.0 / 96.5) < 1e-9


def test_kc_for_pair_flags_a_curve_still_rising_at_the_top(tmp_path):
    info = DRIVER.kc_for_pair({504: [10.0], 8064: [40.0], 64128: [80.0]})  # +100% over the top octave
    assert info["status"] == "rising"
    assert info["kc"] == ">64128"  # lower bound, not a bracket
