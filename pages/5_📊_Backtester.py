import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import warnings
import numpy as np
import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="Portfolio Backtester", page_icon="📊", layout="wide")
st.title("📊 Portfolio Backtester")

# ===================================================
# --- 1. Load Base Data (Bulletproof Version) ---
# ===================================================
@st.cache_data
def load_base_data():
    """
    Loads the price data and benchmark with triple-layer fallback.
    """
    # 1. Load Price Data
    try:
        price_data = pd.read_csv('backtest_price_data.csv', index_col='Date', parse_dates=True)
        # Force timezone-naive to prevent matching errors
        if price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)
    except FileNotFoundError:
        st.error("Error: 'backtest_price_data.csv' not found.")
        return None, None
    
    # 2. Load Benchmark (The Fix)
    nifty_clean = None
    tickers_to_try = ['^NSEI', 'NIFTYBEES.NS'] # Index first, then ETF as backup
    
    for ticker in tickers_to_try:
        try:
            # Download "Max" data to avoid date-range errors
            # auto_adjust=True fixes split/dividend adjustments
            bench = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            
            # Handle Multi-level columns (yfinance update fix)
            if isinstance(bench.columns, pd.MultiIndex):
                try:
                    bench = bench.xs(ticker, axis=1, level=1)
                except:
                    pass # Keep structure if xs fails

            if 'Close' in bench.columns and not bench.empty:
                # Timezone fix
                if bench.index.tz is not None:
                    bench.index = bench.index.tz_localize(None)
                
                # Align dates to our price data
                start_dt = price_data.index.min()
                end_dt = price_data.index.max()
                
                bench_slice = bench['Close'].loc[start_dt:end_dt]
                
                if not bench_slice.empty:
                    nifty_clean = bench_slice.ffill().bfill()
                    break # Success! Stop trying tickers.
                    
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")
            continue

    if nifty_clean is None:
        st.error("⚠️ Comparison Index Unavailable: Could not load NIFTY 50 data.")
    
    return price_data, nifty_clean
# ========================================================
# --- 2. Load Period-Specific Data (Dynamic) ---
# ========================================================
@st.cache_data
def load_period_data(period_key):
    """
    Loads the correct optimization results based on the period_key.
    """
    solutions_file = f'optimized_solutions_{period_key}.csv'
    weights_file = f'optimized_weights_{period_key}.csv'
    
    try:
        solutions_df = pd.read_csv(solutions_file)
        weights_df = pd.read_csv(weights_file)
        
        # Check columns to avoid KeyErrors
        if 'Return' in solutions_df.columns:
            solutions_df['Return_Display'] = solutions_df['Return'] * 100
        if 'CVaR_Risk' in solutions_df.columns:
            solutions_df['CVaR_Risk_Display'] = solutions_df['CVaR_Risk'] * 100
        
        return solutions_df, weights_df
    except FileNotFoundError as e:
        st.error(f"Error: Could not find data file: {e.filename}")
        st.info(f"This period ('{period_key}') does not have data. Did 'main.py' run fully?")
        return None, None

# ==================================================
# --- 3. Backtesting Strategy ---
# ==================================================
def run_simple_backtest(price_data, weights_series, start_date, end_date, cash=100000):
    if weights_series is None or not isinstance(weights_series, pd.Series):
        return None

    # --- Filter data based on user-selected dates ---
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        filtered_price_data = price_data.loc[start_ts:end_ts].copy()
        
        if filtered_price_data.empty or len(filtered_price_data) < 2:
            st.error("Error: Not enough data in the selected date range to run a backtest.")
            return None
    except Exception as e:
        st.error(f"Error filtering dates: {e}")
        return None

    daily_returns = filtered_price_data.pct_change()
    common_stocks = daily_returns.columns.intersection(weights_series.index)
    
    if common_stocks.empty:
        st.error("Error: The selected portfolio contains no stocks present in the price data.")
        return None
        
    weights = weights_series[common_stocks]
    if weights.sum() == 0: return None
    weights = weights / weights.sum()
    
    portfolio_daily_returns = daily_returns[common_stocks].dot(weights)
    portfolio_daily_returns.iloc[0] = 0
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    equity_curve = cash * cumulative_returns
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # --- Calculate dynamic period label ---
    num_days = (filtered_price_data.index[-1] - filtered_price_data.index[0]).days
    period_label = ""
    if num_days > 365 * 1.5:
        period_label = f"({(num_days / 365.25):.1f} Yrs)"
    elif num_days > 60:
        period_label = f"({(num_days / 30.44):.0f} Mos)"
    elif num_days > 1:
        period_label = f"({num_days} Days)"
        
    return {
        "Equity Curve": equity_curve,
        "Total Return [%]": total_return,
        "Max Drawdown [%]": max_drawdown,
        "Period Label": period_label
    }

# ======================================================================
# --- 4.  Backtest Display Function ---
# ======================================================================
def display_backtest_results(stats, name, selected_weights, nifty_benchmark):
    if stats is None:
        st.error("Backtest failed. The selected portfolio may have no valid stocks for the period.")
        return

    st.subheader("Key Performance Metrics")
    period_label = stats.get("Period Label", "")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Portfolio Value", f"₹{stats['Equity Curve'].iloc[-1]:,.0f}")
    col2.metric(f"Total Return {period_label}", f"{stats['Total Return [%]']:.2f}%")
    col3.metric("Max Drawdown", f"{stats['Max Drawdown [%]']:.2f}%")
    
    st.subheader("Portfolio Performance vs. NIFTY 50")
    
    if nifty_benchmark is not None:
        equity_curve = stats['Equity Curve']
        nifty_series_aligned = nifty_benchmark.reindex(equity_curve.index).ffill().bfill()
        
        aligned_df = pd.concat([equity_curve, nifty_series_aligned], axis=1).dropna()
        aligned_df.columns = ["Portfolio", "NIFTY 50"]
        
        if aligned_df.empty or len(aligned_df) < 2:
            st.warning("Could not align NIFTY benchmark data for this short period.")
            return

        initial_cash_value = stats['Equity Curve'].iloc[0]
        nifty_norm = (aligned_df["NIFTY 50"] / aligned_df["NIFTY 50"].iloc[0]) * initial_cash_value
        equity_norm = (aligned_df["Portfolio"] / aligned_df["Portfolio"].iloc[0]) * initial_cash_value
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_norm.index, y=equity_norm.values,
            name=f'{name}', line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=nifty_norm.index, y=nifty_norm.values,
            name='NIFTY 50 Benchmark', line=dict(color='gray', dash='dot', width=2)
        ))
        
        fig.update_layout(title=f"Portfolio Growth (₹{initial_cash_value:,.0f} Initial Investment)", legend_title="Strategy")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Portfolio Allocation Tested")
    st.dataframe(selected_weights[selected_weights > 0.01].sort_values(ascending=False))

# ======================================================================
# --- 5.  Main Page Logic (UPDATED FIX) ---
# ======================================================================
price_data, nifty_benchmark = load_base_data()

if price_data is not None:
    
    st.header("1. Select Backtest Period")
    
    # --- LOGIC: Smart Default Dates ---
    min_date = price_data.index.min().date()
    max_date = price_data.index.max().date()
    
    default_start = min_date 
    is_short_term = False
    
    # Check if coming from a short-term model
    if 'deep_link_period' in st.session_state:
        if st.session_state.deep_link_period in ['3D', '7D', '15D', '30D', '3M']:
            is_short_term = True
            # Set default start to 1 year ago for better chart resolution
            suggested_start = max_date - datetime.timedelta(days=365)
            if suggested_start > min_date:
                default_start = suggested_start

    # --- NOTIFICATION ---
    if is_short_term:
        st.warning(f"⚠️ **Note:** You are testing a short-term strategy ({st.session_state.deep_link_period}). "
                   f"We have auto-set the chart to the last 1 year so you can see recent performance clearly. "
                   f"You can adjust the dates below.")

    st.markdown("Select the time frame you want to run the backtest on.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
    if start_date >= end_date:
        st.error("Error: Start Date must be before End Date.")
        st.stop()

    st.divider()

    # --- CHECKING FOR DEEP LINK ---
    if 'deep_link_index' in st.session_state:
        st.header("2. Backtest Result")
        st.markdown("Running backtest for portfolio selected from the Builder...")
        
        # 1. Get data from session state
        period = st.session_state.deep_link_period
        index = st.session_state.deep_link_index
        name = st.session_state.deep_link_name
        cash_amount = st.session_state.get('deep_link_cash', 100000)
        
        # --- FIX: DO NOT DELETE SESSION STATE HERE ---
        # We keep the variables alive so they survive the page rerun when you change the date.
        
        # 3. Loading the correct data
        _, weights = load_period_data(period)
        
        if weights is not None and index < len(weights):
            selected_weights = weights.iloc[index]
            
            st.subheader(f"Strategy: {name} ({period} Model)")
            with st.spinner(f"Running backtest from {start_date} to {end_date}..."):
                stats = run_simple_backtest(price_data, selected_weights, start_date, end_date, cash=cash_amount)
                display_backtest_results(stats, name, selected_weights, nifty_benchmark)
        else:
            st.error("Could not load portfolio data. Please try again.")
        
        # --- NEW RESET BUTTON ---
        # This is the only way to "Clear" the deep link and go back to manual mode
        if st.button("🔄 Run Another / Reset Backtester", type="secondary", use_container_width=True):
            st.session_state.pop('deep_link_index', None)
            st.session_state.pop('deep_link_name', None)
            st.session_state.pop('deep_link_period', None)
            st.session_state.pop('deep_link_cash', None)
            st.rerun()
            
        st.page_link("Home.py", label="Go to Home Page", icon="🏠")

    # --- NO DEEP LINK: Showing manual dropdowns ---
    else:
        st.header("2. Select Strategy")
        st.markdown("See how your optimized portfolio would have performed in the selected time frame.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            period_list = ['3D', '7D', '15D', '30D', '3M', '6M', '1Y', '3Y', '5Y']
            selected_period = st.selectbox(
                "Model Period:", period_list, index=3,
                help="This will load all portfolios optimized for this time horizon."
            )
        with col2:
            initial_cash = st.number_input(
                "Initial Investment (₹)", 
                min_value=1000, 
                value=100000, 
                step=10000, 
                help="Set the starting cash for the backtest."
            )
        
        solutions, weights = load_period_data(selected_period)
        
        if solutions is not None and weights is not None:
            st.subheader("3. Select Portfolio")
            
            portfolio_map = {}
            try:
                portfolio_map[f"Top Recommended (Max Return)"] = weights.iloc[solutions['Return'].idxmax()]
                portfolio_map[f"Min Risk (CVaR) Portfolio"] = weights.iloc[solutions['CVaR_Risk'].idxmin()]
                portfolio_map[f"Highest ESG Portfolio"] = weights.iloc[solutions['ESG_Score'].idxmax()]
                
                # Check for Cached User Choice
                session_key = f'user_choice_index_{selected_period}'
                if session_key in st.session_state and st.session_state[session_key] is not None:
                    user_index = st.session_state[session_key]
                    if user_index < len(weights):
                        portfolio_map[f"Your Custom Portfolio (Index {user_index})"] = weights.iloc[user_index]
                
                # Check for Cached AI Rec
                ai_rec_key = f'ai_rec_index_{selected_period}'
                if ai_rec_key in st.session_state and st.session_state[ai_rec_key] is not None:
                    ai_index = st.session_state[ai_rec_key]
                    if ai_index < len(weights):
                        portfolio_map[f"AI-Recommended Portfolio (Index {ai_index})"] = weights.iloc[ai_index]

            except Exception as e:
                pass

            selected_portfolio_name = st.selectbox(
                "Choose a portfolio strategy to backtest:",
                list(portfolio_map.keys())
            )
            selected_weights = portfolio_map.get(selected_portfolio_name)
            
            st.subheader("4. Run Backtest")
            
            if st.button(f"Run Backtest for '{selected_portfolio_name}' ({selected_period})", type="primary"):
                if selected_weights is None:
                    st.error("This portfolio is not available.")
                else:
                    with st.spinner(f"Running backtest from {start_date} to {end_date}..."):
                        stats = run_simple_backtest(price_data, selected_weights, start_date, end_date, cash=initial_cash)
                        display_backtest_results(stats, selected_portfolio_name, selected_weights, nifty_benchmark)