# Contributing

Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Create a reproducible development environment:

```bash
uv sync --frozen --extra dev --extra analysis
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Tests use generated data, mocked vendor clients, and temporary directories. They do not require an MT5 terminal or vendor credentials. The test environment clears `DATABENTO_API_KEY` before each test.

For a change to a pipeline stage, exercise both the successful path and the relevant rejection path. Keep schemas and column roles explicit. Changes to feature semantics require a builder version change and an update to [the data contracts](docs/data-contracts.md). Put engineering explanations in the documentation and use descriptive names in code.

Commit `uv.lock` when changing dependencies. Run `uv lock` to update it and verify the frozen environment before opening a pull request. Never commit downloaded market data, credentials, account identifiers, terminal logs, or local run outputs. See [the data policy](docs/data-policy.md).

The CI workflow runs the suite on Linux and Windows with Python 3.11 and 3.13, checks formatting, runs the synthetic demo, and installs the built wheel outside the source checkout.
