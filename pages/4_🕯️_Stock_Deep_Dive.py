import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import talib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import warnings
import datetime
# Suppress warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(
    page_title="Stock Deep Dive",
    page_icon="🕯️",
    layout="wide"
)

# =================================
# --- 1. ALL HELPER FUNCTIONS ---
# =================================
# --- Ticker List Function  ---
@st.cache_data
def get_ticker_list():
    """Loading the list of all available tickers from the source file."""
    try:
        df = pd.read_csv('master_data_for_app.csv', index_col=0)
        tickers = sorted(df.index.unique().tolist())
        return tickers
    except FileNotFoundError:
        st.error("Error: 'master_data_for_app.csv' not found.")
        st.error("Please ensure the 'main_1.py' script has run successfully.")
        return []
    except Exception as e:
        st.error(f"Error in get_ticker_list: {e}")
        return []

# --- Price Data Function (FIXED) ---
@st.cache_data
def get_stock_data_fixed(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        
        # ---  HANDLING YFINANCE MULTIINDEX BUG  ---
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1)
            
        data = data.ffill().bfill().dropna()
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

# --- Feature Creation Function  ---
def create_features_for_explorer(price_df, horizons):
    """
    Using TA-Lib to create a rich set of features for the model.
    NOTE: Assuming price_df has NO NaNs and is float type.
    """
    df = pd.DataFrame(index=price_df.index)
    
    high = price_df['High'].values
    low = price_df['Low'].values
    close = price_df['Close'].values
    volume = price_df['Volume'].values

    df['Close'] = price_df['Close'] 
    
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
    
    for lag in [1, 3, 5]:
        df[f'feature_close_lag_{lag}'] = price_df['Close'].shift(lag)
        df[f'feature_rsi_lag_{lag}'] = df['feature_rsi'].shift(lag)
        df[f'feature_macd_lag_{lag}'] = df['feature_macd'].shift(lag)

    for h in horizons:
        df[f'target_label_{h}d'] = (price_df['Close'].shift(-h) > price_df['Close']).astype(int)
        
    return df

# --- Model Training Function ---
@st.cache_data
def get_model_and_historical_predictions(ticker, horizon_days, start_date='2020-01-02', end_date='2025-11-05'):
    try:
        raw_data = yf.download(ticker, start=start_date, end=end_date)
        if raw_data.empty: return None, None, "Error: Could not download data."
        
        
        # 1. Handling YFinance MultiIndex Bug
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data = raw_data.xs(ticker, axis=1, level=1)

        # 2. Filling all interior holes (like holidays)
        raw_data = raw_data.ffill().bfill()

        # 3. Dropping any remaining NaNs (at the very start/end)
        raw_data = raw_data.dropna()
        
        if raw_data.empty: return None, None, "Error: No data left after cleaning."

        # 4. Explicitly converting all columns to float64 to avert type issues
        raw_data['High'] = raw_data['High'].astype(float)
        raw_data['Low'] = raw_data['Low'].astype(float)
        raw_data['Close'] = raw_data['Close'].astype(float)
        raw_data['Volume'] = raw_data['Volume'].astype(float)
        
        df_featured = create_features_for_explorer(raw_data, horizons=[horizon_days])
        
        target_col = f'target_label_{horizon_days}d'
        feature_cols = [col for col in df_featured.columns if 'feature_' in col]
        
        training_df = df_featured.dropna(subset=[target_col] + feature_cols)
        
        if training_df.empty or len(training_df) < 100:
            return None, None, f"Error: Not enough data to train {horizon_days}-day model."

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
        
        full_feature_set = df_featured[feature_cols].dropna()
        historical_probabilities = model.predict_proba(full_feature_set)[:, 1]
        
        df_final = df_featured.loc[full_feature_set.index].copy()
        df_final['Profit_Probability'] = historical_probabilities
        return df_final, model, None
    except Exception as e:
        st.error(f"An error occurred: {e}") # Show the error
        return None, None, f"An error occurred: {e}"

# --- Main UI Rendering Function ---
def render_model_explorer(period, selected_ticker):
    """Drawing the AI Model Explorer UI."""
    horizon_map = {'3D': 3, '7D': 7, '15D': 15, '30D': 30, '3M': 90}
    horizon_days = horizon_map[period]

    # Pass the error through
    df, model, error_message = get_model_and_historical_predictions(selected_ticker, horizon_days)

    if error_message:
        pass # The error is already displayed by the function
    elif df is not None and model is not None:
        
        # ---  CHART 1 + DESCRIPTION  ---
        st.subheader(f"2a. Price vs. Key Technical Indicators")
        st.markdown("""
        This chart shows the raw data the AI model uses to learn. We chose these indicators because they capture different aspects of a stock's behavior:
        * **Price & SMA:** The closing price (blue) and its 20-Day Simple Moving Average (dotted) show the main trend.
        * **RSI (Relative Strength Index):** The green line in the middle plot is a *momentum* indicator. It measures the speed and change of price movements. A high value (above 70) can signal the stock is "overbought," while a low value (below 30) can signal it's "oversold."
        * **MACD (Moving Average Convergence Divergence):** The red line in the bottom plot is a *trend-following momentum* indicator. When the MACD is above zero, it suggests upward momentum.
        
        The AI's job is to find complex patterns between all these features, something a human eye could never see.
        """)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['feature_sma_20'], name='20-Day SMA', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['feature_rsi'], name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
        fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
        fig.add_trace(go.Scatter(x=df.index, y=df['feature_macd'], name='MACD'), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", row=3, col=1, line_color="gray")
        fig.update_layout(height=600, title_text=f"{selected_ticker} Price & Key Features")
        st.plotly_chart(fig, use_container_width=True)

        # ---  CHART 2 + DESCRIPTION  ---
        st.subheader(f"2b. AI's {period} Profit Probability (Historical)")
        st.markdown(f"""
        This is the **output of the AI model**. For every day in the past, this line shows the model's calculated "Profit Probability" — its confidence that the stock price would be higher **{horizon_days} days** later.
        
        * **Above 0.5 (50%) Line:** The model is "bullish" and predicts a higher price.
        * **Below 0.5 (50%) Line:** The model is "bearish" and predicts a flat or lower price.
        
        This is the *exact* metric we use as the "Return" score in our Short-Term Portfolio Optimizer.
        """)
        fig2 = px.line(df, x=df.index, y='Profit_Probability', title=f"XGBoost Historical {period} Profit Probability")
        fig2.add_hline(y=0.5, line_dash="dot", line_color="red")
        fig2.update_layout(yaxis_title="Profit Probability (0.0 to 1.0)", yaxis_range=[0,1])
        st.plotly_chart(fig2, use_container_width=True)
        
        # ---  CHART 3 + DESCRIPTION  ---
        st.subheader(f"2c. What Features Did the {period} AI Find Important?")
        st.markdown(f"""
        This is the **"Explainable AI" (XAI)** part. It shows us *which* of the dozens of features the model paid the most attention to when making its predictions.
        
        * **Longer Bar = More Important.** A feature with a high importance score was a key driver of the model's predictions (both good and bad).
        * **`feature_close_lag_1`:** This is the stock's price from 1 day ago. It's often the most important, which makes sense!
        * **`feature_rsi`, `feature_adx`:** These are momentum and trend strength indicators.
        
        By looking at this, we can confirm our AI is not "cheating" and is using a healthy mix of trend, momentum, and volume features to make its decisions.
        """)
        importance = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=['Importance'])
        importance = importance.sort_values(by='Importance', ascending=True)
        fig3 = px.bar(importance, x='Importance', y=importance.index, orientation='h', title=f"XGBoost Feature Importance ({period} Model)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
         st.error("Failed to retrieve data and model.")

# ==================================
# --- 2. MAIN PAGE SCRIPT ---
# ==================================

st.title("🕯️ Stock Deep Dive Explorer")
st.markdown("Analyze the historical price data and the AI model's predictions for any stock.")

ticker_list = get_ticker_list()

if ticker_list:
    selected_ticker = st.selectbox(
        "Select a Stock:",
        ticker_list,
        index=ticker_list.index('RELIANCE.NS') if 'RELIANCE.NS' in ticker_list else 0
    )

    if selected_ticker:
        # --- PART 1: Historical Candlestick Chart ---
        st.header(f"1. Historical Price Chart for {selected_ticker}")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-02'))
        with col2:
            end_date = st.date_input("End Date", value=pd.Timestamp.today())

        if start_date < end_date:
            data = get_stock_data_fixed(selected_ticker, start_date, end_date) # Uses the fixed function
            if not data.empty:
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'])]) 
                fig.update_layout(
                    title=f"{selected_ticker} Candlestick Chart",
                    xaxis_title="Date", yaxis_title="Stock Price (INR)",
                    xaxis_rangeslider_visible=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not download price data for {selected_ticker}.")
        else:
            st.error("Error: Start date must be before end date.")
        
        st.divider()

        # --- PART 2: AI Model Explorer  ---
        st.header(f"2. AI Model Analysis for {selected_ticker}")
        
        st.markdown("""
        Select an AI model (time horizon) to see its internal features, historical predictions, and feature importance for this stock. 
        </br>**NOTE:** This explorer is only available for short-term AI models, since long-term models do not use AI predictions.
        """)

        ai_period = st.selectbox(
            "Select AI Model Horizon:",
            options=["3D", "7D", "15D", "30D", "3M"],
            index=3, # Default to 30D
            help="Select the AI model you want to inspect."
        )
        
        # --- Calling the local function ---
        render_model_explorer(ai_period, selected_ticker)
else:
    st.info("Waiting for 'master_data_for_app.csv' to be generated...")