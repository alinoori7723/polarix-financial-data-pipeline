from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from polarix.features.bar_feature_contract import IDENTITY_COLUMNS, MODEL_FEATURE_CANDIDATE_COLUMNS
from polarix.orchestration.artifacts import publish_json


def export_dataset(
    files: list[Path], output_root: Path, *, watermark_ns: int, data_origin: str
) -> tuple[Path, Path]:
    if not files:
        raise ValueError("No feature partitions to export")
    frame = pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")
    frame = frame.filter(
        pl.col("is_model_eligible_candidate")
        & (pl.col("bucket_end_utc_ns") <= watermark_ns)
        & pl.col("bucket_size").is_in(["15s", "60s"])
    ).sort(["symbol_pair", "bucket_size", "feature_timestamp_utc_ns"])
    if frame.is_empty():
        raise ValueError("No eligible, closed feature rows")
    if frame["feature_row_id"].n_unique() != frame.height:
        raise ValueError("Duplicate feature identities")
    feature_columns = list(MODEL_FEATURE_CANDIDATE_COLUMNS)
    for name in feature_columns:
        if not frame.schema[name].is_numeric():
            raise ValueError(f"Feature must be numeric: {name}")
        if frame[name].drop_nulls().is_finite().not_().any():
            raise ValueError(f"Non-finite feature: {name}")
    frame = frame.select(list(IDENTITY_COLUMNS) + feature_columns).with_columns(
        pl.lit(data_origin).alias("data_origin")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    data_path = output_root / "model_features.parquet"
    frame.write_parquet(data_path, compression="zstd")
    contract_path = output_root / "dataset_contract.json"
    publish_json(
        contract_path,
        {
            "schema_version": "1.0",
            "data_origin": data_origin,
            "rows": frame.height,
            "identity_columns": list(IDENTITY_COLUMNS),
            "feature_columns": feature_columns,
            "provenance_columns": ["data_origin"],
            "feature_dtypes": {name: str(frame.schema[name]) for name in feature_columns},
            "feature_availability": "bucket_end_utc_ns",
            "watermark_utc_ns": watermark_ns,
            "null_policy": "retain_warmup_and_undefined_statistics; fit_imputation_on_training_only",
            "label_columns": [],
            "split_policy": "chronological; purge_overlapping_label_horizons_after_label_design",
        },
    )
    return data_path, contract_path


def summarize_dataset(path: Path) -> list[dict]:
    with duckdb.connect(":memory:") as connection:
        cursor = connection.execute(
            """
            SELECT symbol_pair, bucket_size, COUNT(*) AS rows,
                   MIN(feature_timestamp_utc_ns) AS first_feature_utc_ns,
                   MAX(feature_timestamp_utc_ns) AS last_feature_utc_ns,
                   ROUND(AVG(mt5_join_safe_tick_ratio), 6) AS join_safe_ratio
            FROM read_parquet(?)
            GROUP BY symbol_pair, bucket_size
            ORDER BY symbol_pair, bucket_size
            """,
            [str(path)],
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
