import yfinance as yf
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
# 1. Set the Start Date to 2020 (or earlier) to enable long backtests
START_DATE = "2020-01-01" 
END_DATE = datetime.date.today().strftime('%Y-%m-%d')

# 2. Define the file name
OUTPUT_FILE = "backtest_price_data.csv"

def force_update():
    print(f"\n🚀 Starting FORCE UPDATE from {START_DATE} to {END_DATE}...")

    # --- 1. Load Ticker List ---
    # We try to get the list from final_data.csv so it matches your optimized portfolios
    try:
        if os.path.exists("final_data.csv"):
            meta_df = pd.read_csv("final_data.csv")
            # Ensure we are looking at the right column
            if 'Symbol' in meta_df.columns:
                tickers = meta_df['Symbol'].unique().tolist()
                # Append .NS if missing
                tickers = [t + ".NS" if not str(t).endswith(".NS") else t for t in tickers]
            else:
                # Fallback list if CSV structure is different
                print("⚠️ 'Symbol' column not found. Using default NIFTY list.")
                tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        else:
            print("⚠️ final_data.csv not found. Using fallback list.")
            tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
            
    except Exception as e:
        print(f"Error reading ticker list: {e}")
        return

    # --- 2. THE SAFETY FILTER (Crucial) ---
    # We remove the "Poison Pill" tickers that crash the pipeline
    excluded = ['TATAMOTORS.NS']
    tickers = [t for t in tickers if t not in excluded]
    print(f"📋 Downloading data for {len(tickers)} tickers (Excluded: {excluded})...")

    # --- 3. Download Full History ---
    # We download specifically from START_DATE to ensure history is preserved
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=True)
        
        # --- 4. Flatten Multi-Index Columns (Fix for yfinance update) ---
        if isinstance(data.columns, pd.MultiIndex):
            # If we have (Price, Ticker), we just want 'Close' prices usually
            # But for backtesting, we might need OHLC. 
            # If your app expects just 'Close' prices for everything:
            if 'Close' in data.columns:
                data = data['Close']
            
            # If data is still multi-index (Ticker columns), it's fine for simple CSV
            
        # --- 5. Save Directly to the Final File ---
        if not data.empty:
            # Drop columns that are all NaN (failed downloads)
            data = data.dropna(axis=1, how='all')
            
            data.to_csv(OUTPUT_FILE)
            print(f"✅ SUCCESS! Saved {len(data)} rows (from {data.index.min().date()} to {data.index.max().date()}) to '{OUTPUT_FILE}'")
        else:
            print("❌ Error: Download returned empty data.")

    except Exception as e:
        print(f"❌ CRITICAL FAIL: {e}")

if __name__ == "__main__":
    force_update()
