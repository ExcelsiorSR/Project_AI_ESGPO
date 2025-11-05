# --- 1. Import Libraries---
import pandas as pd
import yfinance as yf
import numpy as np
import time
import warnings
import os

# --- AI/ML Imports ---
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import talib # Technical Analysis Library
import xgboost as xgb # Our ML model import
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===========================================================
# --- MODULE 1: XGBoost Predictor (Multi-Horizon)  ---
# ===========================================================

def create_features(price_df, horizons):
    """
     FEATURE UPGRADE: Create a much richer set of features.
    - Adding more advanced indicators (ATR, ADX, OBV).
    - Adding lagged features to give the model a "memory" of recent price action.
    """
    df = pd.DataFrame(index=price_df.index)
    
    # --- Basic OHLCV ---
    high = price_df['High']
    low = price_df['Low']
    close = price_df['Close']
    volume = price_df['Volume']
    
    # --- Advanced Indicators ---
    df['feature_atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['feature_adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['feature_obv'] = talib.OBV(close, volume)
    df['feature_rsi'] = talib.RSI(close, timeperiod=14)
    df['feature_sma_20'] = talib.SMA(close, timeperiod=20)
    df['feature_sma_50'] = talib.SMA(close, timeperiod=50)
    df['feature_mom'] = talib.MOM(close, timeperiod=10)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['feature_macd'] = macd
    df['feature_macdsignal'] = macdsignal
    
    # --- Lagged Features (Giving the model memory) ---
    # Lagging a few key features by 1, 3, and 5 days.
    for lag in [1, 3, 5]:
        df[f'feature_close_lag_{lag}'] = close.shift(lag)
        df[f'feature_rsi_lag_{lag}'] = df['feature_rsi'].shift(lag)
        df[f'feature_macd_lag_{lag}'] = df['feature_macd'].shift(lag)

    # --- Creating a target label for each horizon ---
    for h in horizons:
        df[f'target_label_{h}d'] = (close.shift(-h) > close).astype(int)
        
    return df

def get_predicted_returns(raw_ohlcv_data, valid_tickers, horizons):
    """
    Training 5 separate XGBoost models for each stock for each short-term horizon.
    """
    print("\n--- Starting XGBoost Multi-Horizon Training (CPU Mode) ---")
    
    # This will be a dictionary of dictionaries: { '3d': {ticker: prob, ...}, '7d': ... }
    all_probability_scores = {f"{h}d": {} for h in horizons}
    
    for ticker in valid_tickers:
        print(f"\n  Training models for {ticker}...")
        
        # 1. Creating a dataframe for this stock
        stock_df_full = raw_ohlcv_data.xs(ticker, axis=1, level=1)
        stock_df_featured = create_features(stock_df_full, horizons)
        
        # 2. Looping through and train one model for each horizon
        for h in horizons:
            try:
                target_col = f'target_label_{h}d'
                
                # --- This is the correct pipeline ---
                latest_features = stock_df_featured.iloc[[-1]] # Latest available data point
                training_df = stock_df_featured.dropna(subset=[target_col] + [col for col in stock_df_featured.columns if 'feature_' in col])
                
                if training_df.empty or len(training_df) < 100:
                    print(f"    > [SKIP] Not enough data for {ticker} {h}d model.")
                    all_probability_scores[f"{h}d"][ticker] = 0.5 
                    continue

                feature_cols = [col for col in training_df.columns if 'feature_' in col]
                X = training_df[feature_cols]
                y = training_df[target_col]

                # 3. Building & Training XGBoost Model
                # ---  FEATURE: Hyperparameter Tuning   ---
                # Calculating scale_pos_weight for handling imbalanced classes
                scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1

                # More robust hyperparameters for noisy financial data
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_estimators=200,             # Increased number of trees
                    learning_rate=0.05,           # Slower learning to be more robust
                    max_depth=4,                  # Shallower trees to prevent overfitting
                    subsample=0.8,                # Use 80% of data for each tree
                    colsample_bytree=0.8,         # Use 80% of features for each tree
                    gamma=0.1,                    # Makes the model more conservative
                    scale_pos_weight=scale_pos_weight, # Handles imbalanced data
                    early_stopping_rounds=15,     # Stop if no improvement
                    random_state=42
                )
                # --- END OF FEATURE ---
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # 4. Make Prediction on the *actual* latest features
                latest_features_clean = latest_features[feature_cols]
                probabilities = model.predict_proba(latest_features_clean)
                prob_of_profit = probabilities[0][1] 
                
                all_probability_scores[f"{h}d"][ticker] = prob_of_profit
                print(f"    > {ticker} {h}-Day Profit Probability: {prob_of_profit:.2%}")

            except Exception as e:
                print(f"    > [CRITICAL ERROR] Failed during {ticker} {h}d model: {e}. Skipping.")
                all_probability_scores[f"{h}d"][ticker] = 0.5 
            
    print("--- XGBoost Prediction Complete ---")
    # Converting the dict of dicts into a DataFrame
    return pd.DataFrame(all_probability_scores)

# ======================================================================
# ---  MODULE 2: CVaR Risk Model  ---
# ======================================================================
def calculate_cvar(returns, alpha=0.95):
    """Calculating the 95% Conditional Value at Risk (CVaR)"""
    returns_series = pd.Series(returns)
    var = returns_series.quantile(1 - alpha) # Value at Risk (VaR)
    cvar = returns_series[returns_series <= var].mean()
    return -cvar # Return a positive number for "risk"

# ======================================================================
# --- Data Loading Function  ---
# ======================================================================
def get_master_data(start_date='2020-01-01', end_date='2025-11-01'):
    
    print("\n[Step 2/5] Loading 'final_data.csv'...")
    try:
        esg_df = pd.read_csv('final_data.csv') 
    except FileNotFoundError:
        print("\n[ERROR] 'final_data.csv' not found! Did you upload it?")
        return None, 0, None, None

    # --- Cleaning ESG Data ---
    esg_column_name = 'esg_risk_score_2024'
    esg_df = esg_df[['Symbol', esg_column_name, 'Sector']]
    esg_df = esg_df.dropna(subset=[esg_column_name])
    esg_df['Symbol'] = esg_df['Symbol'].apply(lambda x: x + '.NS')
    esg_df = esg_df.set_index('Symbol')
    tickers = esg_df.index.tolist()
    
    # --- Downloading Financial Data ---
    print("\n[Step 3/5] Downloading 5 years of financial data from yfinance...")
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    # --- Cleaning Data (Imputation Fix) ---
    price_data_filled = raw_data.ffill().bfill()
    price_data_imputed = price_data_filled.dropna(axis=1, how='all')
    
    price_data_cleaned = price_data_imputed['Close']
    clean_tickers = price_data_cleaned.columns.tolist()
    print(f"\nSuccessfully cleaned data for {len(clean_tickers)} stocks.")
    
    # --- Running Predictive AI (XGBoost) ---
    short_term_horizons = [3, 7, 15, 30, 90] # 3d, 7d, 15d, 30d, 3-month
    predicted_returns_xgb = get_predicted_returns(price_data_imputed, clean_tickers, short_term_horizons)

    # --- Calculating Historical Returns (For Long-Term Forecasts) ---
    daily_returns = price_data_cleaned.pct_change().dropna()
    
    hist_returns_6m = daily_returns.iloc[-126:].mean() * 252 
    hist_returns_1y = daily_returns.iloc[-252:].mean() * 252 
    hist_returns_3y = daily_returns.iloc[-756:].mean() * 252 
    hist_returns_5y = daily_returns.mean() * 252 

    # --- Calculating Risk Metric (CVaR based on 5-year data) ---
    cvar_risk = daily_returns.apply(calculate_cvar, axis=0)
    
    # --- Creating "Master" Dataframe ---
    master_df = pd.DataFrame({
        'Return_3d': predicted_returns_xgb['3d'],
        'Return_7d': predicted_returns_xgb['7d'],
        'Return_15d': predicted_returns_xgb['15d'],
        'Return_30d': predicted_returns_xgb['30d'],
        'Return_3m': predicted_returns_xgb['90d'],
        'Return_6m': hist_returns_6m,
        'Return_1y': hist_returns_1y,
        'Return_3y': hist_returns_3y,
        'Return_5y': hist_returns_5y,
        'CVaR_Risk': cvar_risk
    })
    
    master_df = master_df.join(esg_df, how='inner')
    master_df['ESG_Score'] = 100 - master_df[esg_column_name]
    master_df = master_df.drop(columns=[esg_column_name]).dropna() 

    final_tickers = master_df.index.tolist()
    n_stocks = len(final_tickers)
    backtest_data = price_data_cleaned[final_tickers] 

    print("Successfully created all data for optimization and backtesting.")

    return master_df, n_stocks, backtest_data

# ==================================
# --- Optimizer Class ---
# ==================================
class PortfolioProblem(Problem):
    def __init__(self, n_stocks, current_returns, esg_scores, cvar_scores):
        super().__init__(n_var=n_stocks, n_obj=3, n_constr=1, xl=0.0, xu=1.0)
        self.current_returns = current_returns
        self.esg_scores = esg_scores
        self.cvar_scores = cvar_scores

    def _evaluate(self, x, out, *args, **kwargs):
        port_return = -np.dot(x, self.current_returns) 
        port_esg = -np.dot(x, self.esg_scores)
        port_cvar = np.dot(x, self.cvar_scores) 
        sum_of_weights = np.sum(x, axis=1) - 1
        out["F"] = np.column_stack([port_cvar, port_return, port_esg])
        out["G"] = np.column_stack([sum_of_weights])

# ====================================================
# --- MAIN EXECUTION (Multi-Period) ---
# ====================================================
master_df, n_stocks, backtest_data = get_master_data()

if n_stocks > 0:
    
    # Defining ALL 9 periods and their corresponding column name in master_df
    period_columns = {
        '3D': 'Return_3d',
        '7D': 'Return_7d',
        '15D': 'Return_15d',
        '30D': 'Return_30d',
        '3M': 'Return_3m',
        '6M': 'Return_6m',
        '1Y': 'Return_1y',
        '3Y': 'Return_3y',
        '5Y': 'Return_5y'
    }

    # Getting the static risk and ESG data
    esg_scores = master_df['ESG_Score'].values
    cvar_scores = master_df['CVaR_Risk'].values
    
    print(f"\n[Step 4/5] Starting Optimization for all 9 periods...")
    
    for period, return_col in period_columns.items():
        print(f"  Running Optimizer for {period}...")
        
        current_returns = master_df[return_col].values
        
        problem = PortfolioProblem(n_stocks, current_returns, esg_scores, cvar_scores)
        algorithm = NSGA2(pop_size=100)
        termination = get_termination("n_gen", 100)
        
        start_time = time.time()
        res = minimize(problem, algorithm, termination, seed=1, verbose=False) # verbose=False
        print(f"  > Optimization for {period} took {time.time() - start_time:.2f} seconds.")
        
        # --- Processing & Saving Results ---
        final_solutions = pd.DataFrame(res.F, columns=['CVaR_Risk', 'Return', 'ESG_Score'])
        final_solutions['Return'] = -final_solutions['Return']
        final_solutions['ESG_Score'] = -final_solutions['ESG_Score']
        
        # Adding the other columns for the app
        # Calculating Volatility for this portfolio (weighted avg of individual stock vol)
        # This is a simplification, but necessary for a comparable metric
        # Using CVaR as the main risk metric
        final_solutions['Risk (Volatility)'] = final_solutions['CVaR_Risk'] # Use CVaR as the main risk metric
        final_solutions['Prob_Sharpe'] = final_solutions['Return'] / (final_solutions['CVaR_Risk'] + 1e-6) # Use CVaR for Sharpe

        
        final_weights = pd.DataFrame(res.X, columns=master_df.index)
        
        # Save to the new, unique filenames
        solutions_file = f'optimized_solutions_{period}.csv'
        weights_file = f'optimized_weights_{period}.csv'
        
        final_solutions.to_csv(solutions_file, index=False)
        final_weights.to_csv(weights_file, index=False)
        
        print(f"  > Successfully saved '{solutions_file}' and '{weights_file}'.")

    # --- FINAL SAVE AND DOWNLOAD ---
    backtest_data.to_csv('backtest_price_data.csv')
    print("\nSuccessfully saved 'backtest_price_data.csv'.")
    
    
    # Commented out since files.download doesn't work in local scripts(maybe used for cloud)
    # # Download all 18 files + 1 backtest file
    # files.download('backtest_price_data.csv')
    # for period in period_columns.keys():
    #     files.download(f'optimized_solutions_{period}.csv')
    #     files.download(f'optimized_weights_{period}.csv')
    
    print("\n---  FULL LOCAL PIPELINE IS COMPLETE!  ---")
    master_df.to_csv('master_data_for_app.csv')
    print("All result files have been saved to your project directory.")

else:
    print("\nData engineering failed. Cannot proceed.")