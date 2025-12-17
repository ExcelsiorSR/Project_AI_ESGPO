import pandas as pd
import yfinance as yf
import numpy as np
import xgboost as xgb
import talib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# --- 1. Feature Engineering (Same as Main) ---
# ==========================================
def create_features(price_df, horizon):
    df = pd.DataFrame(index=price_df.index)
    
    # Basic data
    close = price_df['Close']
    high = price_df['High']
    low = price_df['Low']
    volume = price_df['Volume']
    
    # Indicators
    df['feature_rsi'] = talib.RSI(close, timeperiod=14)
    df['feature_sma_20'] = talib.SMA(close, timeperiod=20)
    df['feature_adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['feature_mom'] = talib.MOM(close, timeperiod=10)
    macd, _, _ = talib.MACD(close)
    df['feature_macd'] = macd
    
    # Lags
    for lag in [1, 3, 5]:
        df[f'lag_{lag}'] = close.shift(lag)

    # Target: 1 if price in 'horizon' days is higher than today
    df['target'] = (close.shift(-horizon) > close).astype(int)
    
    return df.dropna()

# ==========================================
# --- 2. Data Loading & Audit Function ---
# ==========================================
def audit_model(ticker, horizon_days=3):
    print(f"\n--- Auditing AI Model for {ticker} ({horizon_days}-Day Horizon) ---")
    
    # 1. Download Data
    data = yf.download(ticker, start='2015-01-01', progress=False)
    
    # Fix MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)
        
    data = data.ffill().bfill()
    
    # 2. Create Features
    df = create_features(data, horizon=horizon_days)
    
    # 3. STRICT Time-Based Split (No random shuffling!)
    # Train on first 80%, Test on last 20% (The future)
    split_point = int(len(df) * 0.80)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    feature_cols = [c for c in df.columns if 'feature_' in c or 'lag_' in c]
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    # 4. Train Model
    model = xgb.XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 5. Test Model
    preds = model.predict(X_test)
    
    # 6. Calculate Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    
    print(f"✅ Accuracy:  {acc:.2%} (Baseline: {y_test.mean():.2%})")
    print(f"🎯 Precision: {prec:.2%} (When AI says 'Buy', is it right?)")
    
    return y_test, preds

# ==========================================
# --- 3. Run the Audit ---
# ==========================================
# Test on a stable stock (Reliance) and a volatile one (Tata Motors)
tickers_to_test = ['RELIANCE.NS', 'TATAMOTORS.NS']
horizon = 3 # Testing the 3-Day Sprint model

for t in tickers_to_test:
    y_true, y_pred = audit_model(t, horizon)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Down', 'Pred Up'], yticklabels=['Actual Down', 'Actual Up'])
    plt.title(f"Confusion Matrix: {t} ({horizon}D)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()