from __future__ import annotations

import json
import logging

from speechinsight.logging import get_logger, setup_logging


def test_logs_are_single_line_json(capsys) -> None:
    setup_logging("INFO")
    logger = get_logger("speechinsight.test")
    logger.info("hello", extra={"fields": {"stage": "asr"}})
    captured = capsys.readouterr().err.strip()
    assert "\n" not in captured
    payload = json.loads(captured)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["stage"] == "asr"


def test_debug_below_level_is_suppressed(capsys) -> None:
    setup_logging("WARNING")
    get_logger("speechinsight.test").info("quiet")
    assert capsys.readouterr().err.strip() == ""
    # Reset so later tests are not affected by the raised level.
    logging.getLogger().setLevel(logging.INFO)
