import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="Portfolio Backtester", page_icon="📊", layout="wide")
st.title("📊 Portfolio Backtester")

# ===================================================
# --- 1. Load Base Data (Non-Dynamic) ---
# ===================================================
@st.cache_data
def load_base_data():
    """
    Loads the price data and benchmark, which are static.
    """
    try:
        price_data = pd.read_csv('backtest_price_data.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        st.error("Error: 'backtest_price_data.csv' not found.")
        st.info("Please ensure 'main_1.py' has finished running successfully.")
        return None, None
    
    try:
        nifty = yf.download('^NSEI', start=price_data.index.min(), end=price_data.index.max())
        if nifty.empty:
            st.error("Could not download NIFTY 50 benchmark data.")
            return price_data, None
        nifty_clean = nifty['Close'].ffill().bfill()
    except Exception as e:
        st.error(f"Error downloading NIFTY 50 data: {e}")
        return price_data, None
    
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
        
        solutions_df['Return_Display'] = solutions_df['Return'] * 100
        solutions_df['CVaR_Risk_Display'] = solutions_df['CVaR_Risk'] * 100
        
        return solutions_df, weights_df
    except FileNotFoundError as e:
        st.error(f"Error: Could not find data file: {e.filename}")
        st.info(f"This period ('{period_key}') does not have data. Did 'main_1.py' run fully?")
        return None, None

# ==================================================
# --- 3. Backtesting Strategy (Unchanged) ---
# ==================================================
def run_simple_backtest(price_data, weights_series, cash=100000):
    if weights_series is None or not isinstance(weights_series, pd.Series):
        return None
    daily_returns = price_data.pct_change()
    common_stocks = daily_returns.columns.intersection(weights_series.index)
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
    
    return {
        "Equity Curve": equity_curve,
        "Total Return [%]": total_return,
        "Max Drawdown [%]": max_drawdown
    }

# ======================================================================
# --- 4.  Backtest Display Function  ---
# ======================================================================
def display_backtest_results(stats, name, selected_weights, nifty_benchmark):
    """
    Renders all the metrics and charts for a backtest result.
    """
    if stats is None:
        st.error("Backtest failed. The selected portfolio has no valid stocks.")
        return

    st.subheader("Key Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("End Final Value", f"₹{stats['Equity Curve'].iloc[-1]:,.0f}")
    col2.metric("Total Return (5-Yr)", f"{stats['Total Return [%]']:.2f}%")
    col3.metric("Max Drawdown", f"{stats['Max Drawdown [%]']:.2f}%")
    
    st.subheader("Portfolio Performance vs. NIFTY 50")
    
    if nifty_benchmark is not None:
        equity_curve = stats['Equity Curve']
        nifty_series_aligned = nifty_benchmark.reindex(equity_curve.index).ffill()
        aligned_df = pd.concat([equity_curve, nifty_series_aligned], axis=1).dropna()
        aligned_df.columns = ["Portfolio", "NIFTY 50"]
        
        nifty_norm = (aligned_df["NIFTY 50"] / aligned_df["NIFTY 50"].iloc[0]) * 1_00_000
        equity_norm = (aligned_df["Portfolio"] / aligned_df["Portfolio"].iloc[0]) * 1_00_000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_norm.index, y=equity_norm.values,
            name=f'{name}', line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=nifty_norm.index, y=nifty_norm.values,
            name='NIFTY 50 Benchmark', line=dict(color='gray', dash='dot', width=2)
        ))
        fig.update_layout(title="Portfolio Growth (1 Lakh Invested)", legend_title="Strategy")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Portfolio Allocation Tested")
    st.dataframe(selected_weights[selected_weights > 0.01])

# ======================================================================
# --- 5.  Main Page Logic (Checks for Deep Link)  ---
# ======================================================================
price_data, nifty_benchmark = load_base_data()

if price_data is not None:
    
    # --- CHECKING FOR DEEP LINK ---
    if 'deep_link_index' in st.session_state:
        st.markdown("Running backtest for portfolio selected from the Builder...")
        
        # 1. Get data from session state
        period = st.session_state.deep_link_period
        index = st.session_state.deep_link_index
        name = st.session_state.deep_link_name
        
        # 2. IMPORTANT: Clearing session state so it's a one-time operation
        del st.session_state.deep_link_period
        del st.session_state.deep_link_index
        del st.session_state.deep_link_name

        # 3. Loading the correct data
        _, weights = load_period_data(period)
        
        if weights is not None and index < len(weights):
            selected_weights = weights.iloc[index]
            
            st.header(f"Backtest Result: {name} ({period} Model)")
            with st.spinner("Running backtest..."):
                stats = run_simple_backtest(price_data, selected_weights, cash=1_00_000)
                display_backtest_results(stats, name, selected_weights, nifty_benchmark)
        else:
            st.error("Could not load portfolio data. Please try again.")
        
        st.page_link("pages/5_📊_Backtester.py", label="Run Another Backtest", icon="🔄")
        st.page_link("Home.py", label="Go to Home Page", icon="🏠")

    # --- NO DEEP LINK: Showing manual dropdowns ---
    else:
        st.markdown("See how your optimized portfolio would have *actually* performed over the last 5 years against the NIFTY 50.")
        
        st.header("1. Select Model Period")
        period_list = ['3D', '7D', '15D', '30D', '3M', '6M', '1Y', '3Y', '5Y']
        selected_period = st.selectbox(
            "Model Period:", period_list, index=3,
            help="This will load all portfolios optimized for this time horizon."
        )
        
        solutions, weights = load_period_data(selected_period)
        
        if solutions is not None and weights is not None:
            st.header("2. Select Portfolio Strategy")
            
            portfolio_map = {}
            try:
                portfolio_map[f"Top Recommended (Max Return)"] = weights.iloc[solutions['Return'].idxmax()]
                portfolio_map[f"Min Risk (CVaR) Portfolio"] = weights.iloc[solutions['CVaR_Risk'].idxmin()]
                portfolio_map[f"Highest ESG Portfolio"] = weights.iloc[solutions['ESG_Score'].idxmax()]
                
                session_key = f'user_choice_index_{selected_period}'
                if session_key in st.session_state and st.session_state[session_key] is not None:
                    user_index = st.session_state[session_key]
                    portfolio_map[f"Your Custom Portfolio (Index {user_index})"] = weights.iloc[user_index]
                
                ai_rec_key = f'ai_rec_index_{selected_period}'
                if ai_rec_key in st.session_state and st.session_state[ai_rec_key] is not None:
                    ai_index = st.session_state[ai_rec_key]
                    if ai_index < len(weights):
                        portfolio_map[f"AI-Recommended Portfolio (Index {ai_index})"] = weights.iloc[ai_index]

            except Exception as e:
                st.error(f"Error building portfolio map: {e}")
                st.stop()

            selected_portfolio_name = st.selectbox(
                "Choose a portfolio strategy to backtest:",
                list(portfolio_map.keys())
            )
            selected_weights = portfolio_map.get(selected_portfolio_name)
            
            st.header("3. Run Backtest")
            
            if st.button(f"Run 5-Year Backtest for '{selected_portfolio_name}' ({selected_period})", type="primary"):
                if selected_weights is None:
                    st.error("This portfolio is not available.")
                else:
                    with st.spinner("Running backtest..."):
                        stats = run_simple_backtest(price_data, selected_weights, cash=1_00_000)
                        display_backtest_results(stats, selected_portfolio_name, selected_weights, nifty_benchmark)