# Data Sources

## Primary Provider

**Yahoo Finance** accessed through the open-source `yfinance` library.

- Daily OHLCV bars (Open, High, Low, Close, Volume)
- Default history length: 5 years (`period="5y"`)
- Adjusted prices are used by default (`auto_adjust=True`)
- Results are cached locally as parquet files under `data/raw/`

## Supported International Tickers (verified via yfinance)

| Name     | Ticker | Market |
|----------|--------|--------|
| Apple    | AAPL   | NASDAQ |
| Microsoft| MSFT  | NASDAQ |
| NVIDIA   | NVDA   | NASDAQ |
| Tesla    | TSLA   | NASDAQ |
| Amazon   | AMZN   | NASDAQ |

These symbols are stable and routinely return multi-year history.

## Kenyan Stocks (Nairobi Securities Exchange)

| Name                     | Registry key | Tried symbols      |
|--------------------------|--------------|--------------------|
| Safaricom PLC            | SCOM.NR      | SCOM.NR, SCOM.KE   |
| Equity Group Holdings    | EQTY.NR      | EQTY.NR, EQTY.KE   |
| KCB Group PLC            | KCB.NR       | KCB.NR, KCB.KE     |
| Co-operative Bank        | COOP.NR      | COOP.NR, COOP.KE   |
| Absa Bank Kenya          | ABSA.NR      | ABSA.NR, ABSA.KE   |

### Important notes on Kenyan data

- Yahoo Finance coverage of NSE listings is incomplete and can change without notice.
- StockSight **never invents** prices and **never silently substitutes** an international ticker for a Kenyan one.
- If a download returns empty data, the application raises a clear error and recommends CSV import.
- Users should supply their own verified CSV (Date, Open, High, Low, Close, Volume) when the online provider fails.

## CSV Import Requirements

- File format: CSV
- Required columns (case-insensitive): `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Date values must be parseable by pandas
- Missing prices are **not** filled with zero; rows containing NaNs in OHLCV are dropped

## Validation Rules

- Minimum ~80 observations (enough for 60-step sequences + splits)
- No negative prices or volumes
- Duplicate dates are removed (last occurrence kept)
- Chronological ordering is enforced
- High ≥ Low and Close within High–Low range are checked (warnings only)

## Caching

Successful downloads are stored under `data/raw/` with a companion JSON metadata file recording provider, period, row count and retrieval timestamp. The cache can be bypassed with the “Force refresh” option.
