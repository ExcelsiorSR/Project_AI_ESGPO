import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Short-Term Optimizer",
    page_icon="🚀",
    layout="wide"
)

# ---  Page Title & Intro  ---
st.title("🚀 Short-Term AI Optimizer")
st.markdown("""
Welcome to the **Short-Term Optimizer**. This module is designed for tactical investors with a holding period of 3 days to 3 months.

Each optimizer below uses a **dedicated XGBoost AI model** trained specifically to predict profit probability for that *exact* time horizon.

Please select your desired investment period to begin.
""")

st.subheader("Select a Time Horizon")

col1, col2, col3 = st.columns(3)

# ---   (st.button + st.session_state + st.switch_page)  ---
TARGET_PAGE = "pages/3_🔬_Portfolio_Builder.py"

with col1:
    if st.button("**3-Day Sprint** ⚡", help="Optimized for a 3-day holding period.", use_container_width=True):
        st.session_state.period = "3D"
        st.switch_page(TARGET_PAGE)
    
    if st.button("**30-Day Outlook** 🗓️", help="Our baseline 30-day (1-month) model.", use_container_width=True):
        st.session_state.period = "30D"
        st.switch_page(TARGET_PAGE)
    
with col2:
    if st.button("**7-Day Momentum** 🏃", help="Optimized for a 1-week holding period.", use_container_width=True):
        st.session_state.period = "7D"
        st.switch_page(TARGET_PAGE)

    if st.button("**3-Month Horizon** 📈", help="Optimized for a 3-month (90-day) holding period.", use_container_width=True):
        st.session_state.period = "3M"
        st.switch_page(TARGET_PAGE)

with col3:
    if st.button("**15-Day Swing** 🎯", help="Optimized for a 2-week holding period.", use_container_width=True):
        st.session_state.period = "15D"
        st.switch_page(TARGET_PAGE)

st.divider()

# ---  Preview Section (with index_col=0 fix)  ---
st.markdown("### Preview: Top 5 Stocks (30-Day Model)")
st.markdown("This table shows the 5 stocks with the highest 30-day profit probability as predicted by our XGBoost model.")

try:
    @st.cache_data
    def load_master_data():
        df = pd.read_csv('master_data_for_app.csv', index_col=0)
        return df

    master_df = load_master_data()
    st.dataframe(
        master_df[['Return_30d', 'Return_3m', 'CVaR_Risk', 'ESG_Score']]
        .sort_values(by='Return_30d', ascending=False)
        .head(5)
        .style.format({
            "Return_30d": "{:.2%}",
            "Return_3m": "{:.2%}",
            "CVaR_Risk": "{:.3f}",
            "ESG_Score": "{:.1f}"
        }),
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("Please run `main_1.py` one more time to generate `master_data_for_app.csv` for this preview.")
except KeyError:
    st.error("KeyError: Could not find columns. Make sure 'master_data_for_app.csv' was created correctly.")