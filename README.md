# AI Market Direction Lab

A reproducible machine-learning project for analyzing short-term market direction and evaluating trading signals through historical backtesting.

## Goal

This project estimates the probability that a market ETF will rise over the next five trading days. It is designed for research and education, not financial advice.

## Features

- Downloads historical market data
- Builds technical indicators such as moving averages, RSI, MACD, volatility, and volume features
- Trains classification models to estimate market-direction probability
- Uses walk-forward validation to reduce look-ahead bias
- Compares model performance with simple baseline strategies
- Provides an interactive dashboard for exploring results

## Initial scope

- Default market: Taiwan 50 ETF (`0050.TW`)
- Prediction target: whether the closing price will be higher after five trading days
- Evaluation: accuracy, precision, recall, ROC-AUC, and backtest returns

## Tech stack

Python, pandas, scikit-learn, yfinance, Plotly, and Streamlit.

## Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice, and past performance does not guarantee future results.
