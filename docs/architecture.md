# Architecture and execution model

Polarix has two execution surfaces over shared processing modules:

- `polarix demo` runs a bounded, offline batch in one Python process and gives each execution a new artifact namespace.
- The scripts in `scripts/` expose adapters, quality reports, and the multi-day subprocess orchestrator for separately acquired data.

The demo is the reproducible integration entry point. Its generator emits Arrow-compatible MT5 telemetry and CME-style MBP-1 trade records, which enter the actual normalization and quality implementations.

## Layer responsibilities

| Layer | Responsibility | Persisted evidence |
|---|---|---|
| Ingestion | Generate offline fixtures or capture/read externally supplied events | Bronze Parquet and source metadata |
| Normalization | Convert verified broker timestamps to UTC; preserve event/receive time; bind source schemas | Silver partitions and normalization manifests |
| Quality | Check prices, spreads, residuals, required symbols, schema binding, and reader compatibility | JSON decisions and per-symbol diagnostics |
| Alignment | Evaluate backward event-time matching between CME trades and join-safe MT5 quotes | Match rates, unmatched volume, tolerance diagnostics |
| Aggregation and features | Derive 15s/60s bars, returns, volume ratios, spread statistics, and trailing z-scores | Gold Parquet and feature contract |
| Feature quality and export | Validate column roles, eligibility, finiteness, identity uniqueness, and close watermark | Model feature table and dataset contract |

Polars performs columnar transformations and aggregations. PyArrow supplies explicit storage schemas and Parquet metadata. DuckDB independently reads the resulting files and provides the inspection query; it is not a remote service or a second copy of the dataset.

## Dependencies and failure semantics

The demo graph is an ordered DAG:

```text
ingestion → normalization → quality → alignment
          → aggregation_features → feature_quality → export
```

The runner rejects missing predecessors, duplicate stage names, and invalid ordering before executing any stage. A stage runs only after each named predecessor passes. A failure produces a terminal stage record, and consumers record `SKIPPED` with `UPSTREAM_FAILED` and the blocking predecessors. Independent branches may still run.

Before and after a stage executes, the runner verifies previously published artifacts. Changing an upstream file causes an integrity failure. Failed-stage diagnostics are retained and included in the final inventory. A failed run never authorizes dataset consumption, even when partial files exist.

The multi-day orchestrator uses the same dependency principle with subprocess timeouts, child termination, explicit source-date and run selection, resource checks, and trust checks for existing outputs. Its status vocabulary also distinguishes trusted reuse from untrusted pre-existing files. See [operations](operations.md).

## Immutable metadata and publication

Each demo creates `runs/<run_id>` with exclusive directory creation. Reusing a run ID fails immediately. `metadata.json` records the configuration, configuration hash, package versions, runtime version, and available Git revision/dirty state. Installed wheels may report an unavailable Git revision; the package version remains recorded.

JSON publication writes and flushes a temporary file, then creates a hard link at the final name. Linking fails if that name already exists. Concurrent publication therefore has one winner, without an overwrite window. This requires a filesystem supporting hard links, such as NTFS or a typical Linux local filesystem. Unsupported filesystems fail rather than weakening the rule.

Stage records capture dependencies, timestamps, duration, terminal status, metrics, and output descriptors. Descriptors contain relative paths, byte counts, SHA-256 hashes, and Parquet row counts and schemas. `run_manifest.json` inventories all completed-run artifacts. Inspection validates that inventory before reading the dataset.

This is application-level immutability and integrity checking. The manifest is not signed, and an actor able to replace both a manifest and its artifacts can rewrite local evidence. The design does not claim storage-level WORM guarantees. A process killed before final publication leaves an incomplete run; start a new run ID and preserve the incomplete directory for diagnosis.

## Deliberate tradeoffs

- Local Parquet keeps the demo portable and inspectable. A distributed scheduler and object-store transaction protocol are outside this implementation.
- A strict `PASS` policy keeps the demo simple: `PARTIAL` diagnostics block export. Standalone diagnostic CLIs may return success for `PARTIAL`; inspect their JSON decisions when integrating them.
- Quality evaluation is distinct from feature usefulness. Passing contracts establishes input suitability for further analysis, not predictive value.
- A single stage owns aggregation and feature derivation. Separate materialized aggregates would add a schema and invalidation boundary; they are not required for this bounded batch.
- The fixture window is known complete by construction. Real feeds need acquisition completeness, late-arrival, market calendar, and watermark policies before applying the same export boundary.
