"""Root pytest configuration.

Registers a Hypothesis profile that pins the random seed and skips timing
health checks. Integration tests load heavy ML libraries (torch, speechbrain,
pyannote) that mutate the interpreter's global state in ways that confuse
Hypothesis's data-tree observer on later property tests, producing spurious
``FlakyStrategyDefinition`` errors. Deterministic generation sidesteps the
observer path and keeps the local pytest run as stable as the CI run (where
the ``ml`` marker is deselected)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    database=None,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("default")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run property tests before ``ml``-marked integration tests so Hypothesis
    is not disturbed by heavy ML library import side effects."""

    def sort_key(item: pytest.Item) -> int:
        if "property" in str(item.fspath):
            return 0
        if item.get_closest_marker("ml") is not None:
            return 2
        return 1

    items.sort(key=sort_key)
