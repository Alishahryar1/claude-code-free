from __future__ import annotations

from collections.abc import Iterator

import pytest

from smoke.lib.config import SmokeConfig, auth_headers
from smoke.lib.report import SmokeReport
from smoke.lib.server import RunningServer, start_server


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if SmokeConfig.load().live:
        return
    skip = pytest.mark.skip(reason="set FCC_LIVE_SMOKE=1 to run local smoke tests")
    for item in items:
        item.add_marker(skip)


def pytest_configure(config: pytest.Config) -> None:
    global _REPORT
    smoke_config = SmokeConfig.load()
    _REPORT = SmokeReport(smoke_config)


def pytest_runtest_setup(item: pytest.Item) -> None:
    config = SmokeConfig.load()
    target_marks = list(item.iter_markers("smoke_target"))
    if not target_marks:
        return
    targets = [str(mark.args[0]) for mark in target_marks if mark.args]
    if targets and not any(config.target_enabled(target) for target in targets):
        pytest.skip(f"smoke target disabled: {', '.join(targets)}")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "setup" and not report.skipped:
        return
    if report.when == "teardown" and not report.failed:
        return
    if _REPORT is None:
        return
    markers = sorted(
        str(name) for name in report.keywords if str(name).startswith("smoke_")
    )
    detail = "" if report.longrepr is None else str(report.longrepr)
    _REPORT.add(
        nodeid=report.nodeid,
        outcome=report.outcome,
        duration_s=report.duration,
        markers=markers,
        detail=detail,
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _REPORT is not None:
        _REPORT.write()


@pytest.fixture(scope="session")
def smoke_config() -> SmokeConfig:
    return SmokeConfig.load()


@pytest.fixture
def smoke_server(smoke_config: SmokeConfig) -> Iterator[RunningServer]:
    with start_server(smoke_config) as server:
        yield server


@pytest.fixture
def smoke_headers() -> dict[str, str]:
    return auth_headers()


_REPORT: SmokeReport | None = None
