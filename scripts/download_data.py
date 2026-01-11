import yfinance as yf
import pandas as pd
import os

# Download Bitcoin data
btc = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end="2024-12-31"
)

# 🔥 FINAL FIX: force drop ticker level
btc.columns = btc.columns.droplevel(-1)

# Remove column axis name if exists
btc.columns.name = None

# Create data folder
os.makedirs("data", exist_ok=True)

# Save clean CSV
btc.to_csv("data/btc_usd.csv")

print("Bitcoin dataset downloaded and FULLY CLEANED")
print(btc.head())
print("\nColumns:", list(btc.columns))
