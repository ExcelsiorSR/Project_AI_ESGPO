import yfinance as yf
import pandas as pd
import os
from datetime import datetime

TICKERS = [
    "^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "HINDUNILVR.NS"
]

print(f"🚀 Starting FORCE UPDATE for {len(TICKERS)} tickers...")

try:
    # 2. Download Data with auto_adjust to fix split issues
    print("📉 Downloading from Yahoo Finance...")
    data = yf.download(TICKERS, period="1y", interval="1d", auto_adjust=True, progress=False)
    
    # 3. Flatten Multi-Index Headers (The yfinance fix)
    if isinstance(data.columns, pd.MultiIndex):
        print("🔧 Flattening column headers...")
        # Check if 'Close' exists, otherwise try 'Adj Close'
        try:
            if 'Close' in data.columns.get_level_values(0):
                data = data.xs('Close', level=0, axis=1)
            elif 'Adj Close' in data.columns.get_level_values(0):
                 data = data.xs('Adj Close', level=0, axis=1)
        except:
            pass # Fallback to raw download if structure is simple

    # 4. Clean Date Index
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # 5. Drop empty columns (Clean up failed downloads)
    data = data.dropna(axis=1, how='all')

    # 6. VERIFY DATES
    last_date = data.index[-1].date()
    print(f"📅 Data Ends On: {last_date}")

    if last_date < datetime(2026, 1, 1).date():
        print("❌ CRITICAL ERROR: Downloaded data is still old! Yahoo API might be rate-limiting you.")
    else:
        # 7. Save to V2 Filename (To match your Backtester)
        output_file = "backtest_price_data.csv"
        data.to_csv(output_file)
        print(f"✅ SUCCESS! Saved {len(data)} rows to '{output_file}'")
        print("   (Please check the file manually in VS Code now)")

except Exception as e:
    print(f"❌ Script Failed: {e}")
