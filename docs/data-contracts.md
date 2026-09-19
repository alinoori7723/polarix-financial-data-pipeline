# Data and feature contracts

## Time semantics

| Field | Unit and meaning |
|---|---|
| `time_msc_raw` | MT5 source timestamp in milliseconds; interpreted only with verified session metadata |
| `verified_offset_min` | Broker UTC offset supplied by the selected metadata record |
| `time_msc_utc_ms` | `time_msc_raw - verified_offset_min * 60_000` |
| `recv_time_utc_ms` | Host receive timestamp in UTC milliseconds |
| `residual_ms` | Receive time minus normalized source time |
| `event_time_utc_ns` | CME event time in UTC nanoseconds |
| `receive_time_utc_ns` | CME receive time in UTC nanoseconds |
| `bucket_start_utc_ns` | Inclusive event-time bucket boundary |
| `bucket_end_utc_ns` | Exclusive event-time bucket boundary |
| `feature_timestamp_utc_ns` | Bucket end; earliest event-time interpretation of the completed feature |

Timestamp normalization requires an explicit verified metadata status. An absent or ambiguous broker offset is not inferred from a fixed timezone. Multiple live runs for the same date require explicit selection. The demo's 120-minute offset comes from its synthetic generator contract and is not a statement about a real broker.

The default join-safe residual threshold is 50 ms. Latency outliers fall below -250 ms or above 1,000 ms. Quote validity and spread validity are tracked separately. Inspect the quality report rather than treating a normalized timestamp as proof of clock synchronization.

Tick alignment uses a backward match: a CME event may see only an MT5 quote at or before that event, within the configured tolerance. Alternative tolerances are diagnostic; they do not silently replace the required tolerance.

## Storage and identity

Bronze MT5 Parquet uses the schema in [parquet_writer.py](../src/polarix/ingestion/parquet_writer.py). Silver MT5 adds normalized timestamps, residuals, source metadata, and validity flags. CME normalization records source fields alongside derived trade, side, price, size, and BBO validity. Vendor/schema identifiers describe the adapter format; the demo's run metadata and exported rows explicitly identify data origin as `synthetic`.

Demo paths begin with an exclusive run namespace. Within it, source partitions use symbol and date; raw telemetry also partitions by hour. Gold partitions use symbol pair, date, and bucket size. Metadata records every file consumed or produced inside the run. SHA-256 hashes identify file contents; a row key is not a substitute for a dataset manifest.

Feature row IDs are deterministic UUIDs derived from builder version, date, symbol pair, bucket size, and start timestamp. They are unique within an exported dataset. The same time bucket in different runs can have the same logical row key; run manifests distinguish the datasets and their contents.

## Feature roles and availability

The authoritative allowlist lives in [bar_feature_contract.py](../src/polarix/features/bar_feature_contract.py). It separates identity, quality, absolute-price diagnostics, diagnostic-only features, and model feature candidates. The exporter selects identity and model candidate columns and adds provenance. Targets and future-return labels are not generated.

Candidate groups include returns and ranges, volume ratios, trade intensity and size statistics, quote spread statistics, residual statistics, and normalized cross-market basis. Absolute prices remain in Gold diagnostics and are excluded from the exported model table. The `cme_volume_alignment_ratio` Gold diagnostic is a bar co-presence indicator, not a tick-matched volume estimate; tick match metrics belong to the alignment report.

Returns use the previous bucket in the same symbol pair and bucket size. Z-scores use a trailing window of up to 20 bars, including the current completed bar, with at least two observations. Statistics reset for each daily build. No feature uses the full-day mean or standard deviation. The future-data test extends the input horizon and checks that every earlier closed-bucket feature remains unchanged.

The exporter retains eligible 15s/60s rows whose end timestamp is at or before the supplied watermark. Numeric candidates must be finite when present. Nulls remain for warmup and undefined statistics, such as a zero-variance window. The dataset contract records the ordered feature list and dtypes so consumers need not infer features from every numeric column.

## Training boundary

“Model-ready” means an explicit, numeric feature table with a machine-readable contract. A training pipeline still needs to choose a prediction target, label horizon, sampling policy, estimator-compatible missing-value treatment, and evaluation design.

Use chronological splits. Fit imputation, scaling, feature selection, and other learned transforms on training data only. Purge overlapping label horizons when labels are added. Reset or carry rolling state deliberately at day boundaries. Event-time close timestamps do not prove that all source events had arrived then; live evaluation needs receive-time availability and an explicit late-arrival policy. The synthetic invariance test establishes a bounded causal feature property, not a live backtest.
