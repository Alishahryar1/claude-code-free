from __future__ import annotations

import json
from pathlib import Path

from smoke.lib.report_summary import format_summary, summarize_reports


def test_smoke_readme_uses_env_gated_serial_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "smoke" / "README.md").read_text(encoding="utf-8")

    assert "FCC_LIVE_SMOKE=1" in text
    assert "-n 0" in text
    assert "-m live" not in text


def test_smoke_report_summary_counts_regression_classes(tmp_path: Path) -> None:
    report = {
        "outcomes": [
            {"classification": "missing_env"},
            {"classification": "product_failure"},
            {"classification": "upstream_unavailable"},
        ]
    }
    (tmp_path / "report-one.json").write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_reports(tmp_path)

    assert summary.reports == 1
    assert summary.outcomes == 3
    assert summary.classifications["product_failure"] == 1
    assert summary.has_regression
    assert "status=regression" in format_summary(summary)
