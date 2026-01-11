# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/btc_usd.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

print("✅ Data Loaded Successfully")
print(df.head())

# -----------------------------
# BASIC STATS
# -----------------------------
print("\n--- Descriptive Statistics ---")
print(df.describe())

# -----------------------------
# PLOT CLOSING PRICE
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title('BTC/USD Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# -----------------------------
# MOVING AVERAGES
# -----------------------------
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['MA20'], label='MA20')
plt.plot(df['Date'], df['MA50'], label='MA50')
plt.title('BTC/USD Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# -----------------------------
# SEASONAL DECOMPOSITION
# -----------------------------
df.set_index('Date', inplace=True)
result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
result.plot()
plt.show()

# -----------------------------
# ARIMA FORECAST
# -----------------------------
close_prices = df['Close']

# Fit ARIMA model
model = ARIMA(close_prices, order=(5,1,0))
model_fit = model.fit()
print("\n✅ ARIMA model fitted")

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

# Plot actual + forecast
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Actual')
plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', linestyle='--', color='orange')
plt.title('BTC/USD Closing Price Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# -----------------------------
# SAVE FORECAST
# -----------------------------
forecast_df.to_csv("data/btc_usd_forecast.csv", index=False)
print("✅ Forecast saved to data/btc_usd_forecast.csv")
