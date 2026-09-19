# Operations

## Offline workflow

`polarix demo` creates a new run, writes synthetic Bronze fixtures, executes every required stage, and prints a JSON summary. `--seconds` accepts multiples of 60 between 120 and 3,600. `--seed` controls fixture generation. Omitting `--run-id` creates a unique ID.

`polarix inspect <run-directory>` verifies the artifact inventory and queries the exported table through DuckDB. Exit code `0` means the completed run passed; `2` means pipeline failure, invalid arguments, incomplete output, or integrity failure. Read the JSON error or stage records to identify the cause.

For a rejected or interrupted run, retain the original directory and use a new run ID after correcting the input. There is no in-place resume or overwrite option on the demo command. A missing final manifest is an incomplete run, not a successful run.

## Separately acquired data

The maintained scripts provide explicit paths and a `--help` interface:

| Purpose | Entry point |
|---|---|
| Normalize MT5 telemetry | `scripts/normalize_telemetry.py` |
| Check telemetry | `scripts/telemetry_quality_report.py` |
| Ingest a date-partitioned CME sample | `scripts/ingest_cme_reference_sample.py` |
| Check CME reference data | `scripts/cme_reference_quality_report.py` |
| Measure tick/bar alignment | `scripts/alignment_quality_report.py`, `scripts/bar_alignment_quality_report.py` |
| Build and inspect features | `scripts/build_bar_features.py`, `scripts/bar_feature_quality_report.py` |
| Explore feature distributions | `scripts/feature_eda_report.py` |
| Plan and execute multiple dates | `scripts/build_multiday_plan.py`, `scripts/run_multiday_pipeline.py` |

Defaults use the local `.polarix/` workspace. Pass absolute roots when invoking scripts from another directory. Unlike the demo's exclusive run directories, lower-level tools can offer explicit force/rebuild options. Preserve source data and metadata before using them; trust checks and run selection govern reuse.

The multi-day pipeline records each stage, checks predecessor status, validates trusted existing outputs, constrains source dates, and propagates failures. Resource guards check free disk and available memory. Child processes have timeouts and are tracked for termination on interruption. API retries are bounded. A truncated CME download blocks downstream use unless the caller explicitly selects the documented override; truncation remains recorded in diagnostics.

## Optional source adapters

For Databento dependencies:

```bash
uv sync --frozen --extra databento
uv run --no-sync python scripts/run_multiday_pipeline.py --help
uv run --no-sync python scripts/download_databento_sample.py --help
```

Keep credentials in environment variables. A key alone does not authorize a download: the workflow exposes explicit download, cost acknowledgment, estimate, and physical record-limit gates. A monetary estimate is not a physical cap; review both. The offline demo never enters the vendor download path.

MT5 capture requires Windows, a running terminal, an authorized data source, and the optional `mt5` dependencies. Copy `config/logger.example.json` to `config/logger.config.json` and configure the local paths and symbols. Launch the read-only logger using `polarix-logger --config config/logger.config.json`. The supervisor and Windows wrappers support controlled capture sessions; they are not required by the offline pipeline.

The MT5 adapter exposes telemetry operations and blocks trading operations. Capture preflight, timestamp calibration, resource checks, and periodic kill-switch checks remain separate from successful offline validation. Tests use fake terminal and vendor objects; successful CI is not evidence of a live broker session or licensed vendor entitlement.

## Reproducibility and verification

`uv.lock` pins the development and runtime resolution. CI checks two Python versions on two operating systems, validates synthetic failure paths, and tests a built wheel outside the checkout. Package artifacts and machine-readable test reports are attached to each CI run.

The demo manifest records elapsed stage times as observations of that run. They are not throughput benchmarks. Use the synthetic stress scripts for bounded local diagnostics, and define representative workload, hardware, memory budget, and acceptance criteria before making scaling claims.
