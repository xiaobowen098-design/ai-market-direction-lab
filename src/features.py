"""Create technical indicators and a future market-direction target."""

import pandas as pd


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add technical features and a five-trading-day direction target."""
    data = prices.copy()

    close = data["Close"]
    volume = data["Volume"]

    # Moving averages and relative position.
    data["return_1d"] = close.pct_change()
    data["ma_5"] = close.rolling(window=5).mean()
    data["ma_20"] = close.rolling(window=20).mean()
    data["ma_5_vs_20"] = data["ma_5"] / data["ma_20"] - 1

    # RSI using 14 trading days.
    daily_change = close.diff()
    average_gain = daily_change.clip(lower=0).rolling(window=14).mean()
    average_loss = -daily_change.clip(upper=0).rolling(window=14).mean()
    relative_strength = average_gain / average_loss
    data["rsi_14"] = 100 - (100 / (1 + relative_strength))

    # MACD.
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    # Volatility and relative volume.
    data["volatility_20"] = data["return_1d"].rolling(window=20).std()
    average_volume_20 = volume.rolling(window=20).mean()
    data["volume_vs_20d_avg"] = volume / average_volume_20

    # Target: 1 if the adjusted close is higher five trading days later.
    future_close = close.shift(-5)
    data["target_up_5d"] = (future_close > close).where(future_close.notna())

    # Remove rows without enough history or without a future target.
    data = data.dropna().copy()
    data["target_up_5d"] = data["target_up_5d"].astype(int)
    return data
