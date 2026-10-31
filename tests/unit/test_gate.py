from __future__ import annotations

from pathlib import Path

from speechinsight.evaluation.gate import check_file, check_results


def test_smoke_fixture_passes_the_gate() -> None:
    """The committed KPI smoke fixture must pass the gate — it is the one CI
    will run against on every PR."""

    verdict = check_file(Path("tests/fixtures/kpi_smoke.json"))
    assert verdict.passed, verdict.summary()


def test_wer_above_threshold_fails() -> None:
    data = {"wer": {"points": [{"split": "test-clean", "model": "base.en", "wer": 0.09}]}}
    verdict = check_results(data)
    assert not verdict.passed
    assert any("WER" in f for f in verdict.failures)


def test_der_above_threshold_fails() -> None:
    verdict = check_results({"der": {"der": 0.40}})
    assert not verdict.passed
    assert any("DER" in f for f in verdict.failures)


def test_schema_too_low_fails() -> None:
    verdict = check_results({"extraction": {"schema_validity": 0.90}})
    assert not verdict.passed
    assert any("schema" in f.lower() for f in verdict.failures)


def test_unsupported_entities_too_high_fails() -> None:
    verdict = check_results({"extraction": {"unsupported_entities_share": 0.10}})
    assert not verdict.passed
    assert any("unsupported entities" in f.lower() for f in verdict.failures)


def test_missing_fields_do_not_fail_the_gate() -> None:
    # An empty report has no fields to check — the gate must stay silent.
    assert check_results({}).passed


def test_dev_split_is_not_gated_on_test_thresholds() -> None:
    data = {
        "wer": {
            "points": [
                {"split": "dev-clean", "model": "base.en", "wer": 0.20},  # bad, but dev
            ]
        }
    }
    assert check_results(data).passed


def test_summary_format() -> None:
    assert check_results({}).summary() == "KPI gate: PASS"
    bad = check_results({"der": {"der": 0.50}})
    assert bad.summary().startswith("KPI gate: FAIL")
