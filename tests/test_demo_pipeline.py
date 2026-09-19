from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polarix.cli import main
from polarix.features.bar_feature_contract import (
    ABSOLUTE_PRICE_NON_FEATURE_COLUMNS,
    MODEL_FEATURE_CANDIDATE_COLUMNS,
)
from polarix.ingestion.fixtures import FixtureConfig
from polarix.orchestration.artifacts import ArtifactIntegrityError, sha256_file
from polarix.orchestration.demo import inspect_run, run_demo


def test_end_to_end_reads_real_parquet_and_exports_only_contract_features(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path / "analyst's workspace", FixtureConfig(seconds=120), "example")
    summary = inspect_run(run_dir)
    assert summary["status"] == "PASS"
    assert summary["integrity"] == "VERIFIED"
    assert [table["rows"] for table in summary["feature_tables"]] == [8, 2, 8, 2]
    frame = pl.read_parquet(run_dir / "dataset" / "model_features.parquet")
    contract = json.loads((run_dir / "dataset" / "dataset_contract.json").read_text())
    assert contract["feature_columns"] == list(MODEL_FEATURE_CANDIDATE_COLUMNS)
    assert not set(ABSOLUTE_PRICE_NON_FEATURE_COLUMNS).intersection(frame.columns)
    assert frame["data_origin"].unique().to_list() == ["synthetic"]
    assert frame["feature_row_id"].n_unique() == 20
    assert (frame["feature_timestamp_utc_ns"] == frame["bucket_end_utc_ns"]).all()
    assert contract["label_columns"] == []
    assert frame["cme_return_close_to_close"].null_count() == 4


@pytest.mark.parametrize(
    "scenario,failed_stage",
    [
        ("invalid-timestamps", "normalization"),
        ("crossed-quotes", "quality"),
    ],
)
def test_invalid_inputs_persist_failure_and_block_feature_publication(
    tmp_path: Path, scenario: str, failed_stage: str
) -> None:
    run_dir = run_demo(tmp_path, FixtureConfig(seconds=120, scenario=scenario), scenario)
    summary = inspect_run(run_dir)
    assert summary["status"] == "FAIL"
    failed_index = next(
        index for index, stage in enumerate(summary["stages"]) if stage["name"] == failed_stage
    )
    assert summary["stages"][failed_index]["status"] == "FAIL"
    assert all(stage["status"] == "SKIPPED" for stage in summary["stages"][failed_index + 1 :])
    assert not (run_dir / "dataset").exists()
    assert not (run_dir / "gold").exists()


def test_seed_and_row_id_reproducibility(tmp_path: Path) -> None:
    first = run_demo(tmp_path, FixtureConfig(seconds=120), "first")
    second = run_demo(tmp_path, FixtureConfig(seconds=120), "second")
    assert sha256_file(first / "dataset" / "model_features.parquet") == sha256_file(
        second / "dataset" / "model_features.parquet"
    )
    for path in (first / "bronze").rglob("*.parquet"):
        assert sha256_file(path) == sha256_file(second / path.relative_to(first))


def test_future_ticks_do_not_change_features_for_closed_earlier_buckets(tmp_path: Path) -> None:
    short = run_demo(tmp_path, FixtureConfig(seconds=120), "short")
    long = run_demo(tmp_path, FixtureConfig(seconds=240), "long")
    before = pl.read_parquet(short / "dataset" / "model_features.parquet")
    after = pl.read_parquet(long / "dataset" / "model_features.parquet")
    after = after.filter(pl.col("bucket_end_utc_ns") <= FixtureConfig(seconds=120).end_ns)
    assert_frame_equal(before, after)


def test_immutable_run_id_refuses_overwrite(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path, FixtureConfig(seconds=120), "same")
    previous = sha256_file(run_dir / "run_manifest.json")
    with pytest.raises(FileExistsError):
        run_demo(tmp_path, FixtureConfig(seconds=180), "same")
    assert sha256_file(run_dir / "run_manifest.json") == previous


def test_inspection_detects_changed_artifact(tmp_path: Path) -> None:
    run_dir = run_demo(tmp_path, FixtureConfig(seconds=120), "tamper")
    path = run_dir / "dataset" / "model_features.parquet"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="Artifact changed"):
        inspect_run(run_dir)


@pytest.mark.parametrize("run_id", ["../escape", "CON", "a/b", "", "x" * 65])
def test_run_namespace_validation(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        run_demo(tmp_path, FixtureConfig(seconds=120), run_id)


def test_cli_returns_nonzero_for_blocked_run(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "demo",
            "--output",
            str(tmp_path),
            "--run-id",
            "failed",
            "--seconds",
            "120",
            "--scenario",
            "invalid-timestamps",
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
