"""Download historical price data for the project."""

import pandas as pd
import yfinance as yf


def download_history(
    ticker: str = "0050.TW",
    period: str = "5y",
) -> pd.DataFrame:
    """Download adjusted daily prices and return them as a DataFrame."""
    prices = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if prices.empty:
        raise ValueError(
            f"No price data returned for {ticker}. Check the ticker or try again later."
        )

    # Handle yfinance versions that return grouped, multi-level column names.
    if isinstance(prices.columns, pd.MultiIndex):
        if ticker in prices.columns.get_level_values(-1):
            prices = prices.xs(ticker, axis=1, level=-1)
        else:
            prices.columns = prices.columns.get_level_values(0)

    prices = prices.dropna(subset=["Close"])
    return prices


if __name__ == "__main__":
    history = download_history()
    print(history.tail())
