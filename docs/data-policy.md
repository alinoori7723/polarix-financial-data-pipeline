# Data policy

The repository distributes source code and synthetic examples. The generator creates invented price paths, trade sizes, event times, and quote spreads. These records are not downloaded, anonymized, sampled, or derived from a licensed market-data feed.

Real CME, broker, and vendor datasets must remain outside version control unless their applicable license explicitly permits redistribution. A small sample is still subject to its license. The MIT source-code license does not grant rights to third-party data or trademarks.

Local `.polarix/` output, data and report directories, Parquet, DBN, archives, environment files, and local logger configuration are ignored. Ignore rules prevent common mistakes but do not authorize distribution or prevent force-adding a file. Inspect staged changes before publishing.

Do not commit account identifiers, access keys, terminal screenshots, private documents, broker logs, or machine-specific exports. The example configuration contains only sandbox defaults. Credentials belong in the environment or an external secret store.

For any additional distributable fixture, record its synthetic generator or explicit redistribution permission, origin, schema, and intended use. Keep examples small and deterministic. All feature counts and quality results in the demo refer to synthetic data and do not establish market performance.
