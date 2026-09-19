from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.quality.bar_feature_quality import (
    QualityConfig,
    build_feature_quality_report,
    write_reports,
)

DEFAULT_FEATURES_ROOT = Path(".polarix/data/features/bar_features")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix feature quality report")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = QualityConfig(
        date=args.date, features_root=Path(args.features_root), reports_root=Path(args.reports_root)
    )
    report = build_feature_quality_report(cfg)
    json_path, txt_path = write_reports(cfg, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "decision_reason": report.get("decision_reason"),
                "json_report": str(json_path),
                "txt_report": str(txt_path),
            },
            indent=2,
        )
    )
    return 0 if report["decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
