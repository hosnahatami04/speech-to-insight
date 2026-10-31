"""KPI smoke-gate: parse a tiny results.json and check it against the
thresholds in spec §4. The real evaluation produces a real results.json;
CI only runs this gate against a committed fixture so the pipeline shape
is validated on every change without the heavy ML runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GateVerdict:
    passed: bool
    failures: tuple[str, ...]

    def summary(self) -> str:
        if self.passed:
            return "KPI gate: PASS"
        return "KPI gate: FAIL\n" + "\n".join(f"  - {f}" for f in self.failures)


# Thresholds come from spec §4. "report" entries exist in the structure but do
# not have a numeric target.
_DER_MAX = 0.35
_WER_SMALL_MAX = 0.05
_WER_BASE_MAX = 0.07
_SCHEMA_MIN = 0.98
_EVIDENCE_REQUIRED = 1.0
# dropped-items share is reported (spec §4.3) but not gated on — the "healthy"
# direction depends on the mix of extractor behaviour and grounding strictness.
_UNSUPPORTED_ENTITIES_MAX = 0.02
_NLI_MIN = 0.90


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check_results(results: dict[str, Any]) -> GateVerdict:
    """Validate a results dict against the KPI thresholds. Missing fields are
    not failures — the gate only fires on values that are present AND outside
    the threshold, so partial runs stay useful."""

    failures: list[str] = []

    for entry in _get(results, "wer", "points", default=[]) or []:
        if entry.get("split") != "test-clean":
            continue
        model = entry.get("model", "")
        wer = entry.get("wer")
        if wer is None:
            continue
        if "small" in model and wer > _WER_SMALL_MAX:
            failures.append(f"WER test-clean {model} = {wer:.3f} > {_WER_SMALL_MAX}")
        if "base" in model and wer > _WER_BASE_MAX:
            failures.append(f"WER test-clean {model} = {wer:.3f} > {_WER_BASE_MAX}")

    der = _get(results, "der", "der")
    if der is not None and der > _DER_MAX:
        failures.append(f"DER = {der:.3f} > {_DER_MAX}")

    schema = _get(results, "extraction", "schema_validity")
    if schema is not None and schema < _SCHEMA_MIN:
        failures.append(f"schema validity = {schema:.3f} < {_SCHEMA_MIN}")

    evidence = _get(results, "extraction", "evidence_valid_share")
    if evidence is not None and evidence < _EVIDENCE_REQUIRED:
        failures.append(f"evidence valid share = {evidence:.3f} < {_EVIDENCE_REQUIRED}")

    unsupported_entities = _get(results, "extraction", "unsupported_entities_share")
    if unsupported_entities is not None and unsupported_entities > _UNSUPPORTED_ENTITIES_MAX:
        failures.append(
            f"unsupported entities share = {unsupported_entities:.3f} > {_UNSUPPORTED_ENTITIES_MAX}"
        )

    nli = _get(results, "extraction", "nli_supported_share")
    if nli is not None and nli < _NLI_MIN:
        failures.append(f"NLI supported share = {nli:.3f} < {_NLI_MIN}")

    return GateVerdict(passed=not failures, failures=tuple(failures))


def check_file(path: str | Path) -> GateVerdict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return check_results(data)
