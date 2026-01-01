import streamlit as st

st.set_page_config(
    page_title="Home - AI ESG Optimizer",
    page_icon="🌿",
    layout="wide"
)

st.title("Welcome to the AI-Driven ESG Portfolio Optimizer 🌿")

st.markdown("""
This tool is a full-stack, multi-module AI system for modern, responsible investing in the **NIFTY 50**.

It uses **XGBoost** for short-term predictions and **historical analysis** for long-term strategies, all while optimizing for Risk (CVaR) and ESG scores using an **Evolutionary Algorithm (NSGA-II)**.
""")

st.subheader("Choose Your Investment Style")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>🚀 Short-Term Tactical</h3>", unsafe_allow_html=True)
        st.markdown("For investors with a **3-day to 3-month** horizon. Uses AI (XGBoost) to predict profit probability.")
        if st.button("Go to Short-Term Optimizer", icon="🚀", use_container_width=True):
            st.switch_page("pages/1_🚀_Short_Term_Optimizer.py")


with col2:
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>⏳ Long-Term Strategic</h3>", unsafe_allow_html=True)
        st.markdown("For investors with a **6-month to 5-year** horizon. Uses historical annualized returns.")
        if st.button("Go to Long-Term Optimizer", icon="⏳", use_container_width=True):
            st.switch_page("pages/2_⏳_Long_Term_Optimizer.py")

st.divider()

# ---  DISCLAIMER BLOCK  ---
st.warning(
    "**Disclaimer: For Educational Use Only**\n\n"
    "This tool is a term project and is intended for educational and illustrative purposes only. "
    "It provides a statistical approach to calculate probable returns and **is not an investment advice.** "
    "All financial investments carry significant risk. You must critically evaluate the risks involved with a qualified financial advisor and adhere to all SEBI guidelines before making any investment decisions.\n "
    "\nFor official guidance, please review the [SEBI Investor Dos and Don'ts](https://investor.sebi.gov.in/securities-dos_and_donts.html)."
)
# ---  END OF BLOCK  ---

st.divider()

st.subheader("Other Tools")

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Stock Deep Dive & AI Explorer", icon="🕯️", use_container_width=True):
        st.switch_page("pages/4_🕯️_Stock_Deep_Dive.py")
        
with col_b:
    if st.button("Portfolio Backtester", icon="📊", use_container_width=True):
        st.switch_page("pages/5_📊_Backtester.py")
