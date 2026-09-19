from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from polarix.orchestration.artifacts import (
    ArtifactIntegrityError,
    artifact_record,
    publish_json,
    verify_artifacts,
)
from polarix.orchestration.runner import Stage, StageOutput, execute_stages


def test_failed_dependency_blocks_consumer_but_independent_branch_runs(tmp_path: Path) -> None:
    def fail() -> StageOutput:
        raise RuntimeError("source rejected")

    def forbidden() -> StageOutput:
        pytest.fail("Blocked stage ran")

    records = execute_stages(
        tmp_path,
        (
            Stage("source", (), fail),
            Stage("dependent", ("source",), forbidden),
            Stage("independent", (), lambda: StageOutput(metrics={"completed": True})),
        ),
    )
    assert [record["status"] for record in records] == ["FAIL", "SKIPPED", "PASS"]
    assert records[1]["blocked_by"] == ["source"]
    assert records[2]["metrics"]["completed"]


def test_stage_cannot_mutate_an_upstream_artifact(tmp_path: Path) -> None:
    path = tmp_path / "source.json"

    def source() -> StageOutput:
        publish_json(path, {"value": 1})
        return StageOutput((path,))

    def corrupt() -> StageOutput:
        path.write_text("changed")
        return StageOutput()

    records = execute_stages(
        tmp_path,
        (
            Stage("source", (), source),
            Stage("transform", ("source",), corrupt),
            Stage("consumer", ("transform",), lambda: pytest.fail("Corrupt lineage consumed")),
        ),
    )
    assert records[1]["status"] == "FAIL"
    assert records[1]["reason"] == "ArtifactIntegrityError"
    assert records[2]["status"] == "SKIPPED"


@pytest.mark.parametrize(
    "stages",
    [
        (Stage("bad", ("missing",), lambda: StageOutput()),),
        (Stage("same", (), lambda: StageOutput()), Stage("same", (), lambda: StageOutput())),
        (Stage("../escape", (), lambda: StageOutput()),),
    ],
)
def test_invalid_graph_rejected_before_any_stage_runs(
    tmp_path: Path, stages: tuple[Stage, ...]
) -> None:
    with pytest.raises(ValueError):
        execute_stages(tmp_path, stages)
    assert not (tmp_path / "stages").exists()


def test_atomic_metadata_publication_has_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"

    def publish(index: int) -> bool:
        try:
            publish_json(path, {"writer": index})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(publish, range(8)))
    assert sum(results) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_paths_cannot_escape_run_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    publish_json(outside, {"secret": "fixture"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = artifact_record(outside, tmp_path)
    record["path"] = "../outside.json"
    with pytest.raises(ArtifactIntegrityError, match="escapes run directory"):
        verify_artifacts(run_dir, [record])
