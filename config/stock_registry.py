"""Stock registry for international and Kenyan equities.

Yahoo Finance ticker symbols are used where verified.
Kenyan stocks on the Nairobi Securities Exchange (NSE) often use
suffixes such as .NR or .KE. Availability can change; CSV import
is the recommended fallback when yfinance returns empty data.
"""

from typing import Dict, List, Optional, Any


STOCK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # International (US major exchanges) - well supported by yfinance
    "AAPL": {
        "name": "Apple Inc.",
        "market": "NASDAQ",
        "ticker": "AAPL",
        "currency": "USD",
        "provider_symbol": "AAPL",
        "country": "United States",
        "data_available": True,
        "verified": True,
        "notes": "Primary data via yfinance",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "market": "NASDAQ",
        "ticker": "MSFT",
        "currency": "USD",
        "provider_symbol": "MSFT",
        "country": "United States",
        "data_available": True,
        "verified": True,
        "notes": "Primary data via yfinance",
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "market": "NASDAQ",
        "ticker": "NVDA",
        "currency": "USD",
        "provider_symbol": "NVDA",
        "country": "United States",
        "data_available": True,
        "verified": True,
        "notes": "Primary data via yfinance",
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "market": "NASDAQ",
        "ticker": "TSLA",
        "currency": "USD",
        "provider_symbol": "TSLA",
        "country": "United States",
        "data_available": True,
        "verified": True,
        "notes": "Primary data via yfinance",
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "market": "NASDAQ",
        "ticker": "AMZN",
        "currency": "USD",
        "provider_symbol": "AMZN",
        "country": "United States",
        "data_available": True,
        "verified": True,
        "notes": "Primary data via yfinance",
    },
    # Kenyan stocks (Nairobi Securities Exchange)
    # Common yfinance symbols for NSE include .NR or .KE.
    # These are registered with best-known symbols; availability must be
    # checked at runtime. CSV import is always available as fallback.
    "SCOM.NR": {
        "name": "Safaricom PLC",
        "market": "NSE",
        "ticker": "SCOM",
        "currency": "KES",
        "provider_symbol": "SCOM.NR",
        "country": "Kenya",
        "data_available": None,  # checked at runtime
        "verified": False,
        "notes": "Try SCOM.NR or SCOM.KE. Prefer CSV if yfinance empty.",
        "alternative_symbols": ["SCOM.KE", "SCOM"],
    },
    "EQTY.NR": {
        "name": "Equity Group Holdings",
        "market": "NSE",
        "ticker": "EQTY",
        "currency": "KES",
        "provider_symbol": "EQTY.NR",
        "country": "Kenya",
        "data_available": None,
        "verified": False,
        "notes": "Try EQTY.NR or EQTY.KE. Prefer CSV if yfinance empty.",
        "alternative_symbols": ["EQTY.KE", "EQTY"],
    },
    "KCB.NR": {
        "name": "KCB Group PLC",
        "market": "NSE",
        "ticker": "KCB",
        "currency": "KES",
        "provider_symbol": "KCB.NR",
        "country": "Kenya",
        "data_available": None,
        "verified": False,
        "notes": "Try KCB.NR or KCB.KE. Prefer CSV if yfinance empty.",
        "alternative_symbols": ["KCB.KE", "KCB"],
    },
    "COOP.NR": {
        "name": "Co-operative Bank of Kenya",
        "market": "NSE",
        "ticker": "COOP",
        "currency": "KES",
        "provider_symbol": "COOP.NR",
        "country": "Kenya",
        "data_available": None,
        "verified": False,
        "notes": "Try COOP.NR or COOP.KE. Prefer CSV if yfinance empty.",
        "alternative_symbols": ["COOP.KE", "COOP"],
    },
    "ABSA.NR": {
        "name": "Absa Bank Kenya PLC",
        "market": "NSE",
        "ticker": "ABSA",
        "currency": "KES",
        "provider_symbol": "ABSA.NR",
        "country": "Kenya",
        "data_available": None,
        "verified": False,
        "notes": "Try ABSA.NR or ABSA.KE. Prefer CSV if yfinance empty.",
        "alternative_symbols": ["ABSA.KE", "ABSA"],
    },
}


def get_stock_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Return registry entry for a symbol (case-insensitive)."""
    key = symbol.upper().strip()
    if key in STOCK_REGISTRY:
        return STOCK_REGISTRY[key].copy()
    # Also search by short ticker
    for info in STOCK_REGISTRY.values():
        if info.get("ticker", "").upper() == key:
            return info.copy()
        if info.get("provider_symbol", "").upper() == key:
            return info.copy()
    return None


def list_stocks(market: Optional[str] = None, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return list of registered stocks, optionally filtered."""
    results = []
    for symbol, info in STOCK_REGISTRY.items():
        entry = info.copy()
        entry["registry_key"] = symbol
        if market and entry.get("market", "").upper() != market.upper():
            continue
        if country and entry.get("country", "").upper() != country.upper():
            continue
        results.append(entry)
    return results


def list_tickers() -> List[str]:
    """Return list of provider symbols."""
    return [info["provider_symbol"] for info in STOCK_REGISTRY.values()]


def get_display_name(symbol: str) -> str:
    """Human-readable name for a symbol."""
    info = get_stock_info(symbol)
    if info:
        return f"{info['name']} ({info['ticker']})"
    return symbol
