# Security Policy

## Supported versions

StockSight is maintained on the `main` branch. Security fixes are applied there and released as
needed.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Report a vulnerability**.
3. Describe the issue, affected files or versions, and reproduction steps.

You can expect an acknowledgement within a few days. Please allow reasonable time for a fix before
public disclosure.

## Scope and design notes

StockSight runs locally as a single-user educational application:

- No authentication or multi-tenancy is included. Do not expose the Streamlit server to untrusted
  networks without adding authentication and TLS.
- Outbound network access is limited to Yahoo Finance for market data.
- SQL statements use parameterized queries.
- Uploaded CSVs are validated, and file names used for exports or caches are sanitized.
- PyTorch checkpoints are loaded with `weights_only=True`.
- No credentials or secrets are stored in the repository. Local overrides belong in
  `.streamlit/secrets.toml`, which is git-ignored.

## Not in scope

- Forecast accuracy or investment outcomes (see the disclaimer in the README).
- Vulnerabilities in third-party packages that are already fixed upstream.
