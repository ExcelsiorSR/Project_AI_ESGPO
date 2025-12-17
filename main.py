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
    Creates technical indicators and lagged features.
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
    
    # --- Lagged Features ---
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
    Trains XGBoost models and calculates BOTH Probability AND Expected Return.
    """
    print("\n--- Starting XGBoost Multi-Horizon Training (Expected Return Mode) ---")
    
    # We now store TWO values for every stock/horizon
    all_expected_returns = {f"{h}d": {} for h in horizons}
    all_probabilities = {f"{h}d": {} for h in horizons}
    
    for ticker in valid_tickers:
        print(f"\n  Training models for {ticker}...")
        
        # 1. Prepare Data
        stock_df_full = raw_ohlcv_data.xs(ticker, axis=1, level=1)
        stock_df_featured = create_features(stock_df_full, horizons)
        
        for h in horizons:
            try:
                # --- NEW LOGIC: Calculate Historical Win/Loss Size ---
                # "When this stock moves over 'h' days, how big is the move?"
                hist_pct_change = stock_df_full['Close'].pct_change(periods=h)
                
                # Avg Win: Mean of positive returns
                avg_win = hist_pct_change[hist_pct_change > 0].mean()
                # Avg Loss: Absolute mean of negative returns
                avg_loss = abs(hist_pct_change[hist_pct_change <= 0].mean())
                
                # Fallbacks for stability (if stock is flat or new)
                if pd.isna(avg_win): avg_win = 0.02  # Default 2%
                if pd.isna(avg_loss): avg_loss = 0.02 # Default 2%

                # --- Train Model ---
                target_col = f'target_label_{h}d'
                latest_features = stock_df_featured.iloc[[-1]] 
                
                training_df = stock_df_featured.dropna(subset=[target_col] + [col for col in stock_df_featured.columns if 'feature_' in col])
                
                if training_df.empty or len(training_df) < 100:
                    print(f"    > [SKIP] Not enough data for {ticker} {h}d.")
                    all_expected_returns[f"{h}d"][ticker] = 0.0
                    all_probabilities[f"{h}d"][ticker] = 0.5
                    continue

                feature_cols = [col for col in training_df.columns if 'feature_' in col]
                X = training_df[feature_cols]
                y = training_df[target_col]

                scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1

                model = xgb.XGBClassifier(
                    objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                    scale_pos_weight=scale_pos_weight, early_stopping_rounds=15, random_state=42
                )
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # --- Prediction ---
                latest_features_clean = latest_features[feature_cols]
                prob_of_profit = model.predict_proba(latest_features_clean)[0][1] 
                
                # --- NEW LOGIC: Expected Value Calculation ---
                # EV = (Prob_Win * Avg_Win) - (Prob_Loss * Avg_Loss)
                expected_return = (prob_of_profit * avg_win) - ((1 - prob_of_profit) * avg_loss)
                
                all_expected_returns[f"{h}d"][ticker] = expected_return
                all_probabilities[f"{h}d"][ticker] = prob_of_profit
                
                print(f"    > {ticker} {h}d | Prob: {prob_of_profit:.0%} | Exp. Return: {expected_return:.2%}")

            except Exception as e:
                print(f"    > [ERROR] {ticker} {h}d failed: {e}")
                all_expected_returns[f"{h}d"][ticker] = 0.0
                all_probabilities[f"{h}d"][ticker] = 0.5
            
    print("--- XGBoost Training Complete ---")
    return pd.DataFrame(all_expected_returns), pd.DataFrame(all_probabilities)

# ======================================================================
# ---  MODULE 2: CVaR Risk Model  ---
# ======================================================================
def calculate_cvar(returns, alpha=0.95):
    """Calculating the 95% Conditional Value at Risk (CVaR)"""
    returns_series = pd.Series(returns)
    var = returns_series.quantile(1 - alpha)
    cvar = returns_series[returns_series <= var].mean()
    return -cvar 

# ======================================================================
# --- Data Loading Function  ---
# ======================================================================
def get_master_data(start_date='2020-01-01', end_date='2025-11-01'):
    
    print("\n[Step 2/5] Loading 'final_data.csv'...")
    try:
        esg_df = pd.read_csv('final_data.csv') 
    except FileNotFoundError:
        print("\n[ERROR] 'final_data.csv' not found!")
        return None, 0, None, None

    esg_column_name = 'esg_risk_score_2024'
    esg_df = esg_df[['Symbol', esg_column_name, 'Sector']]
    esg_df = esg_df.dropna(subset=[esg_column_name])
    esg_df['Symbol'] = esg_df['Symbol'].apply(lambda x: x + '.NS')
    esg_df = esg_df.set_index('Symbol')
    tickers = esg_df.index.tolist()
    
    print("\n[Step 3/5] Downloading 5 years of financial data from yfinance...")
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    price_data_filled = raw_data.ffill().bfill()
    price_data_imputed = price_data_filled.dropna(axis=1, how='all')
    price_data_cleaned = price_data_imputed['Close']
    clean_tickers = price_data_cleaned.columns.tolist()
    print(f"\nSuccessfully cleaned data for {len(clean_tickers)} stocks.")
    
    # --- Running Predictive AI (XGBoost) ---
    short_term_horizons = [3, 7, 15, 30, 90] 
    
    # Get BOTH Expected Returns (for Math) AND Probabilities (for Display)
    expected_returns_xgb, probs_xgb = get_predicted_returns(price_data_imputed, clean_tickers, short_term_horizons)

    # --- Calculating Historical Returns (Long-Term) ---
    daily_returns = price_data_cleaned.pct_change().dropna()
    hist_returns_6m = daily_returns.iloc[-126:].mean() * 252 
    hist_returns_1y = daily_returns.iloc[-252:].mean() * 252 
    hist_returns_3y = daily_returns.iloc[-756:].mean() * 252 
    hist_returns_5y = daily_returns.mean() * 252 

    # --- Calculating Risk Metric ---
    cvar_risk = daily_returns.apply(calculate_cvar, axis=0)
    
    # --- Creating "Master" Dataframe ---
    master_df = pd.DataFrame({
        # Returns for Optimizer (Now using Expected Returns for short term)
        'Return_3d': expected_returns_xgb['3d'],
        'Return_7d': expected_returns_xgb['7d'],
        'Return_15d': expected_returns_xgb['15d'],
        'Return_30d': expected_returns_xgb['30d'],
        'Return_3m': expected_returns_xgb['90d'],
        'Return_6m': hist_returns_6m,
        'Return_1y': hist_returns_1y,
        'Return_3y': hist_returns_3y,
        'Return_5y': hist_returns_5y,
        
        # Storing Probabilities for UI Display later
        'Prob_3d': probs_xgb['3d'],
        'Prob_7d': probs_xgb['7d'],
        'Prob_15d': probs_xgb['15d'],
        'Prob_30d': probs_xgb['30d'],
        'Prob_3m': probs_xgb['90d'],
        
        'CVaR_Risk': cvar_risk
    })
    
    master_df = master_df.join(esg_df, how='inner')
    master_df['ESG_Score'] = 100 - master_df[esg_column_name]
    master_df = master_df.drop(columns=[esg_column_name]).dropna() 

    final_tickers = master_df.index.tolist()
    n_stocks = len(final_tickers)
    backtest_data = price_data_cleaned[final_tickers] 

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
        # We maximize Return (minimize negative return)
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
    
    # Mapping for Short-Term (Expected Return + Probability)
    # Mapping for Long-Term (Historical Return + No Prob)
    
    period_config = {
        '3D':  {'ret': 'Return_3d',  'prob': 'Prob_3d'},
        '7D':  {'ret': 'Return_7d',  'prob': 'Prob_7d'},
        '15D': {'ret': 'Return_15d', 'prob': 'Prob_15d'},
        '30D': {'ret': 'Return_30d', 'prob': 'Prob_30d'},
        '3M':  {'ret': 'Return_3m',  'prob': 'Prob_3m'},
        '6M':  {'ret': 'Return_6m',  'prob': None}, # No AI for long term
        '1Y':  {'ret': 'Return_1y',  'prob': None},
        '3Y':  {'ret': 'Return_3y',  'prob': None},
        '5Y':  {'ret': 'Return_5y',  'prob': None}
    }

    esg_scores = master_df['ESG_Score'].values
    cvar_scores = master_df['CVaR_Risk'].values
    
    print(f"\n[Step 4/5] Starting Optimization for all 9 periods...")
    
    for period, config in period_config.items():
        print(f"  Running Optimizer for {period}...")
        
        return_col = config['ret']
        prob_col = config['prob']
        
        current_returns = master_df[return_col].values
        
        problem = PortfolioProblem(n_stocks, current_returns, esg_scores, cvar_scores)
        algorithm = NSGA2(pop_size=100)
        termination = get_termination("n_gen", 100)
        
        start_time = time.time()
        res = minimize(problem, algorithm, termination, seed=1, verbose=False)
        print(f"  > Optimization for {period} took {time.time() - start_time:.2f} seconds.")
        
        # --- Processing Results ---
        final_solutions = pd.DataFrame(res.F, columns=['CVaR_Risk', 'Return', 'ESG_Score'])
        
        # Invert back to positive numbers
        final_solutions['Return'] = -final_solutions['Return']
        final_solutions['ESG_Score'] = -final_solutions['ESG_Score']
        
        # --- NEW: Calculate Portfolio Probability Score (Weighted Avg) ---
        final_weights = pd.DataFrame(res.X, columns=master_df.index)
        
        if prob_col is not None:
            # For Short Term: Calculate weighted AI Probability
            # dot product of Weights matrix and Probability Vector
            probs_vector = master_df[prob_col].values
            final_solutions['Prob_Score'] = np.dot(res.X, probs_vector)
        else:
            # For Long Term: Just fill with 0 or NaN
            final_solutions['Prob_Score'] = 0.0

        # Calculate Sharpe (using Expected Return / Risk)
        final_solutions['Prob_Sharpe'] = final_solutions['Return'] / (final_solutions['CVaR_Risk'] + 1e-6)

        # Save files
        solutions_file = f'optimized_solutions_{period}.csv'
        weights_file = f'optimized_weights_{period}.csv'
        
        final_solutions.to_csv(solutions_file, index=False)
        final_weights.to_csv(weights_file, index=False)
        
        print(f"  > Saved '{solutions_file}' and '{weights_file}'.")

    # --- FINAL SAVE ---
    backtest_data.to_csv('backtest_price_data.csv')
    master_df.to_csv('master_data_for_app.csv')
    print("\n---  FULL LOCAL PIPELINE IS COMPLETE!  ---")
    print("All result files have been saved.")

else:
    print("\nData engineering failed. Cannot proceed.")