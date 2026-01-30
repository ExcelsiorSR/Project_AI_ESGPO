import yfinance as yf
import pandas as pd
import os
import subprocess
import sys

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = "backtest_price_data.csv"
MAIN_SCRIPT = "main.py"

# List of tickers to ensure we cover your portfolio
# (Includes Nifty 50 and major stocks to be safe)
TICKERS = [
    "^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "HINDUNILVR.NS", "TATAMOTORS.NS"
]

def clean_and_download_data():
    print(f"📉 Downloading fresh data for {len(TICKERS)} tickers...")
    try:
        # 1. Download Max Data
        data = yf.download(TICKERS, period="max", auto_adjust=True, progress=False)
        
        # 2. FLATTEN HEADERS (The Critical Fix)
        # New yfinance returns MultiIndex columns. We need to drop the 'Price' level.
        if isinstance(data.columns, pd.MultiIndex):
            print("🔧 Detected Multi-Level Columns. Flattening...")
            # If 'Close' is a level, select it. Otherwise, just drop the top level.
            try:
                if 'Close' in data.columns.get_level_values(0):
                     data = data.xs('Close', level=0, axis=1)
                elif 'Adj Close' in data.columns.get_level_values(0):
                     data = data.xs('Adj Close', level=0, axis=1)
                else:
                    # Fallback: Just keep the ticker level (usually level 1)
                    data.columns = data.columns.get_level_values(1)
            except Exception as e:
                # Last resort flatten
                data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]

        # 3. Ensure Index is Datetime and Sorted
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # 4. Remove Timezone info (Fixes comparison errors)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # 5. Save Clean CSV
        data.to_csv(DATA_FILE)
        print(f"✅ Success! Saved clean data to {DATA_FILE}")
        print(f"📅 Latest Date: {data.index[-1]}")
        
        return True

    except Exception as e:
        print(f"❌ Data Download Failed: {e}")
        return False

def run_optimizer():
    print(f"🚀 Running {MAIN_SCRIPT} to update portfolios...")
    try:
        # Run main.py using the same python environment
        result = subprocess.run([sys.executable, MAIN_SCRIPT], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Optimization Complete!")
            print(result.stdout[-500:]) # Print last 500 chars of output
        else:
            print(f"⚠️ {MAIN_SCRIPT} finished with errors (might be okay if it just saved files).")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Failed to run optimizer: {e}")

if __name__ == "__main__":
    print("--- STARTING FINAL FIX ---")
    if clean_and_download_data():
        run_optimizer()
    print("--- DONE. NOW RESTART STREAMLIT ---")