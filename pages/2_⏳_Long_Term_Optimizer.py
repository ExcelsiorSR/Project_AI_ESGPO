import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Long-Term Optimizer",
    page_icon="⏳",
    layout="wide"
)

# ---  Page Title & Intro  ---
st.title("⏳ Long-Term AI Optimizer")
st.markdown("""
Welcome to the **Long-Term Optimizer**. This module is for strategic investors with a holding period of 6 months to 5 years.

These optimizers use **annualized historical returns** as the "Return" metric, focusing on stable, long-term performance.

Please select your desired investment horizon to begin.
""")

st.subheader("Select a Time Horizon")

col1, col2 = st.columns(2)

# ---  (st.button + st.session_state + st.switch_page)  ---
TARGET_PAGE = "pages/3_🔬_Portfolio_Builder.py"

with col1:
    if st.button("**6-Month Outlook** 📈", help="Optimized using 6-month historical annualized returns.", use_container_width=True):
        st.session_state.period = "6M"
        st.switch_page(TARGET_PAGE)
    
    if st.button("**3-Year Plan** 🗓️", help="Optimized using 3-year historical annualized returns.", use_container_width=True):
        st.session_state.period = "3Y"
        st.switch_page(TARGET_PAGE)
    
with col2:
    if st.button("**1-Year Strategy** 🎯", help="Optimized using 1-year historical annualized returns.", use_container_width=True):
        st.session_state.period = "1Y"
        st.switch_page(TARGET_PAGE)

    if st.button("**5-Year Vision** 🏦", help="Optimized using 5-year historical annualized returns.", use_container_width=True):
        st.session_state.period = "5Y"
        st.switch_page(TARGET_PAGE)

st.divider()

# ---  Preview Section (with index_col=0 fix)  ---
st.markdown("### Preview: Top 5 Stocks (Long-Term Models)")
st.markdown("This table shows the 5 stocks with the highest 5-year annualized historical returns from our data.")

try:
    @st.cache_data
    def load_master_data():
        df = pd.read_csv('master_data_for_app.csv', index_col=0)
        return df

    master_df = load_master_data()
    
    st.dataframe(
        master_df[['Return_1y', 'Return_5y', 'CVaR_Risk', 'ESG_Score']]
        .sort_values(by='Return_5y', ascending=False)
        .head(5)
        .style.format({
            "Return_1y": "{:.2%}", # Formatted as percentage
            "Return_5y": "{:.2%}", # Formatted as percentage
            "CVaR_Risk": "{:.3f}",
            "ESG_Score": "{:.1f}"
        }),
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("Preview data not found. Please ensure `main_1.py` has run successfully and created `master_data_for_app.csv`.")
except KeyError:
    st.error("KeyError: Could not find columns. Make sure 'master_data_for_app.csv' was created correctly.")