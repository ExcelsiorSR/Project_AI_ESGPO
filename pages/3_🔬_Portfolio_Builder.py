import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Builder",
    page_icon="🔬",
    layout="wide"
)

# ===========================================
# --- 1. DYNAMIC PERIOD DEFINITIONS ---
# ===========================================

# This dictionary maps the URL parameter to the correct files and labels
PERIOD_DEFINITIONS = {
    "3D": {
        "title": "3-Day Sprint",
        "return_label": "AI Profit Probability",
        "is_ai": True
    },
    "7D": {
        "title": "7-Day Momentum",
        "return_label": "AI Profit Probability",
        "is_ai": True
    },
    "15D": {
        "title": "15-Day Swing",
        "return_label": "AI Profit Probability",
        "is_ai": True
    },
    "30D": {
        "title": "30-Day Outlook",
        "return_label": "AI Profit Probability",
        "is_ai": True
    },
    "3M": {
        "title": "3-Month Horizon",
        "return_label": "AI Profit Probability",
        "is_ai": True
    },
    "6M": {
        "title": "6-Month Strategy",
        "return_label": "Annualized Return",
        "is_ai": False
    },
    "1Y": {
        "title": "1-Year Strategy",
        "return_label": "Annualized Return",
        "is_ai": False
    },
    "3Y": {
        "title": "3-Year Plan",
        "return_label": "Annualized Return",
        "is_ai": False
    },
    "5Y": {
        "title": "5-Year Vision",
        "return_label": "Annualized Return",
        "is_ai": False
    },
}

# --- Getting the period from SESSION STATE ---
# This is set by the buttons on the welcome pages
period = st.session_state.get("period", "30D") # Default to 30D
if period not in PERIOD_DEFINITIONS:
    period = "30D"

CURRENT_PERIOD = PERIOD_DEFINITIONS[period]
RETURN_LABEL = CURRENT_PERIOD["return_label"]
IS_AI_MODEL = CURRENT_PERIOD["is_ai"]

# --- Helper Function for CSV Download ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# ======================================
# --- 2. DYNAMIC DATA LOADING ---
# ======================================
@st.cache_data
def load_data(period_key):
    """
    Loads the correct optimization results based on the period_key.
    """
    solutions_file = f'optimized_solutions_{period_key}.csv'
    weights_file = f'optimized_weights_{period_key}.csv'
    master_data_file = 'master_data_for_app.csv' # Our new sector/info source
    
    try:
        solutions_df = pd.read_csv(solutions_file)
        weights_df = pd.read_csv(weights_file)
        master_df = pd.read_csv(master_data_file, index_col=0) # <-- Fixed this!
    except FileNotFoundError as e:
        st.error(f"Error: Could not find data file: {e.filename}")
        st.error(f"Please ensure `main_1.py` has run successfully and all 'optimized_..._{period_key}.csv' files are present.")
        return None, None, None

    # --- Create Sector Map ---
    sector_map = master_df['Sector'] # <-- Fixed this!
    
    # --- Data Cleaning (from main_1.py) ---
    solutions_df['Return_Display'] = solutions_df['Return'] * 100
    solutions_df['CVaR_Risk_Display'] = solutions_df['CVaR_Risk'] * 100
    
    return solutions_df, weights_df, sector_map

# --- Load the Data ---
solutions, weights, sector_map = load_data(period)

# --- Define Tooltips (now dynamic) ---
help_return = f"{RETURN_LABEL} (0-100). For AI models, this is the profit probability. For long-term models, this is the annualized historical return. (Higher is better)"
help_risk = "Conditional Value at Risk (CVaR) (0-100). Measures the expected loss in a 'tail risk' scenario. (Lower is better)"
help_esg = "ESG Score (0-100) based on Environmental, Social, and Governance ratings. (Higher is better)"
help_sharpe = f"{RETURN_LABEL} / CVaR Risk. A custom 'reward-for-risk' ratio. (Higher is better)"

# ===================================
# --- 3. MAIN APP (Dynamic) ---
# ===================================
if solutions is not None:
    
    st.title(f"🔬 {CURRENT_PERIOD['title']} Optimizer")

    # ====================================================
    # ---  PART 1: The 'Best' Overall Portfolio  ---
    # ====================================================
    st.header("1. Our Top Recommended Portfolio")
    st.markdown(f"Based on our analysis, this is the single portfolio on the {period} frontier with the **Highest {RETURN_LABEL}**.")
    
    top_portfolio = solutions.loc[solutions['Return'].idxmax()]
    top_portfolio_index = top_portfolio.name

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"📈 {RETURN_LABEL}", f"{top_portfolio['Return_Display']:.2f}%", help=help_return)
    col2.metric("📉 CVaR Risk", f"{top_portfolio['CVaR_Risk_Display']:.2f}%", help=help_risk)
    col3.metric("🏆 Sharpe Ratio", f"{top_portfolio['Prob_Sharpe']:.2f}", help=help_sharpe)
    col4.metric("🌿 ESG Score", f"{top_portfolio['ESG_Score']:.2f}", help=help_esg)

    with st.expander("Show Top Portfolio Allocation"):
        best_weights_series = weights.iloc[top_portfolio_index]
        best_weights_df = pd.DataFrame(best_weights_series[best_weights_series > 0.01]) # Filter > 1%
        best_weights_df.columns = ['Weight']
        best_weights_df = best_weights_df.merge(sector_map, left_index=True, right_index=True)
        
        csv_data = convert_df_to_csv(best_weights_df[['Weight', 'Sector']].sort_values(by='Weight', ascending=False))
        st.download_button(
            label="📥 Download Allocation as CSV",
            data=csv_data,
            file_name=f"Top_Portfolio_{period}.csv",
            mime='text/csv',
        )

        col_pie, col_tree = st.columns(2)
        with col_pie:
            fig_pie = px.pie(
                best_weights_df, values='Weight', names=best_weights_df.index,
                title="Allocation by Stock"
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="top_pie_chart")

        with col_tree:
            fig_tree = px.treemap(
                best_weights_df, path=[px.Constant("Portfolio"), 'Sector', best_weights_df.index],
                values='Weight', title="Allocation by Sector and Stock"
            )
            st.plotly_chart(fig_tree, use_container_width=True, key="top_tree_chart")

    st.divider()

    # ================================================
    # ---  PART 2: The Interactive Screener  ---
    # ================================================
    st.header("2. Customise Your Own Portfolio")
    st.markdown("Use the sliders to filter the optimal portfolios to find the ones that match your goals.")

    min_ret, max_ret = float(solutions['Return_Display'].min()), float(solutions['Return_Display'].max())
    min_risk, max_risk = float(solutions['CVaR_Risk_Display'].min()), float(solutions['CVaR_Risk_Display'].max())
    min_esg, max_esg = float(solutions['ESG_Score'].min()), float(solutions['ESG_Score'].max())

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Profit Goal")
        user_min_ret = st.slider(
            f"Minimum {RETURN_LABEL} (%)",
            min_value=min_ret, max_value=max_ret,
            value=min_ret, step=0.1, format="%.2f", help=help_return
        )
    with col2:
        st.subheader("Risk Tolerance")
        user_max_risk = st.slider(
            "Maximum CVaR Risk (%)",
            min_value=min_risk, max_value=max_risk,
            value=max_risk, step=0.1, format="%.2f", help=help_risk
        )
    with col3:
        st.subheader("Ethical Priority")
        user_min_esg = st.slider(
            "Minimum ESG Score",
            min_value=min_esg, max_value=max_esg,
            value=min_esg, step=0.1, format="%.1f", help=help_esg
        )

    # --- Filtering the Data ---
    filtered_solutions = solutions[
        (solutions['Return_Display'] >= user_min_ret) &
        (solutions['CVaR_Risk_Display'] <= user_max_risk) &
        (solutions['ESG_Score'] >= user_min_esg)
    ].copy() 

    st.subheader("Filtered Optimal Portfolios")

    if filtered_solutions.empty:
        st.warning("No portfolio on the optimal frontier matches your exact criteria. Try loosening your constraints.")
        st.session_state[f'user_choice_index_{period}'] = None # Clear any previous choice for this period
    else:
        st.markdown(f"Found **{len(filtered_solutions)}** portfolios that meet your goals:")
        
        display_df = filtered_solutions.copy()
        display_df[RETURN_LABEL] = display_df['Return_Display'].apply(lambda x: f"{x:.2f}%")
        display_df['CVaR_Risk'] = display_df['CVaR_Risk_Display'].apply(lambda x: f"{x:.2f}%")
        display_df['ESG_Score'] = display_df['ESG_Score'].apply(lambda x: f"{x:.1f}")
        display_df['Sharpe_Ratio'] = display_df['Prob_Sharpe'].apply(lambda x: f"{x:.2f}")
        
        selected_row = st.dataframe(
            display_df[[RETURN_LABEL, 'CVaR_Risk', 'Sharpe_Ratio', 'ESG_Score']],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if selected_row.selection["rows"]:
            selected_index = filtered_solutions.index[selected_row.selection["rows"][0]]
            st.session_state[f'user_choice_index_{period}'] = selected_index
            
            st.subheader(f"Allocation for Selected Portfolio (Index {selected_index})")

            selected_weights_series = weights.iloc[selected_index]
            selected_weights_df = pd.DataFrame(selected_weights_series[selected_weights_series > 0.01])
            selected_weights_df.columns = ['Weight']
            selected_weights_df = selected_weights_df.merge(sector_map, left_index=True, right_index=True)

            csv_data_custom = convert_df_to_csv(selected_weights_df[['Weight', 'Sector']].sort_values(by='Weight', ascending=False))
            st.download_button(
                label="📥 Download This Allocation as CSV",
                data=csv_data_custom,
                file_name=f"Custom_Portfolio_{period}_{selected_index}.csv",
                mime='text/csv',
            )

            col_pie, col_tree = st.columns(2)
            with col_pie:
                fig_pie_cust = px.pie(
                    selected_weights_df, values='Weight', names=selected_weights_df.index,
                    title="Allocation by Stock"
                )
                st.plotly_chart(fig_pie_cust, use_container_width=True, key="custom_pie_chart")
            with col_tree:
                fig_tree_cust = px.treemap(
                    selected_weights_df, path=[px.Constant("Portfolio"), 'Sector', selected_weights_df.index],
                    values='Weight', title="Allocation by Sector"
                )
                st.plotly_chart(fig_tree_cust, use_container_width=True, key="custom_tree_chart")
        else:
            st.info("Click on a row in the table above to see its allocation.")
            st.session_state[f'user_choice_index_{period}'] = None

    st.divider()
    
    # =================================================================
    # --- 🚀 PART 3: The AI Recommendation 🚀 ---
    # =================================================================
    st.header("3. Our Recommendation")
    
    # --- Defining ALL keys at the top ---
    session_key = f'user_choice_index_{period}'
    ai_rec_key = f'ai_rec_index_{period}' 

    if session_key not in st.session_state or st.session_state[session_key] is None:
        st.info("Select a portfolio from the table in Part 2 to get a custom recommendation.")
        st.session_state[ai_rec_key] = None # Clearing any old recommendation
    else:
        user_choice_index = st.session_state[session_key]
        user_choice = solutions.loc[user_choice_index]
        initial_cash = st.number_input(
        "Initial Investment (₹)", 
        value=100000, 
        step=10000, 
        key="cash_input_global",
        help="Set the starting cash for the backtest."
        )
        
        risk_tolerance = user_choice['CVaR_Risk_Display'] * 1.05 
        esg_tolerance = user_choice['ESG_Score'] * 0.95
        
        better_portfolios = solutions[
            (solutions.index != user_choice.name) & 
            (solutions['Return_Display'] > user_choice['Return_Display']) & 
            (solutions['CVaR_Risk_Display'] <= risk_tolerance) & 
            (solutions['ESG_Score'] >= esg_tolerance)
        ]
        
        if better_portfolios.empty:
            st.success("🎉 **Excellent Choice!**")
            st.markdown(f"The portfolio you've selected (Portfolio **{user_choice.name}**) is a top-tier choice. Our AI could not find another portfolio with a **higher {RETURN_LABEL}** that *also* maintained your approximate Risk and ESG levels.")
            st.session_state[ai_rec_key] = None # No recommendation to save
            
            # ---  ADDING BACKTEST BUTTON FOR USER'S CHOICE ---
            if st.button(f"📊 Backtest Your Choice (Portfolio {user_choice.name})", use_container_width=True, type="primary"):
                st.session_state.deep_link_period = period
                st.session_state.deep_link_index = user_choice.name
                st.session_state.deep_link_name = f"Your Choice (Portfolio {user_choice.name})"
                st.session_state.deep_link_cash = initial_cash
                st.switch_page("pages/5_📊_Backtester.py")
            # --- End of button ---

        else:
            st.success("😃 **A Better Option May Exist!**")
            st.markdown(f"Your selected portfolio is good, but our AI found *at least one* other portfolio with a **higher {RETURN_LABEL}** while staying close to your chosen risk and ESG levels.")
            
            our_recommendation = better_portfolios.loc[better_portfolios['Return_Display'].idxmax()]
            
            # --- Saving the index of the recommendation ---
            st.session_state[ai_rec_key] = our_recommendation.name 
            
            st.subheader("Comparison")
            col_user, col_ai = st.columns(2)
            
            with col_user:
                st.markdown(f"**Your Choice (Portfolio {user_choice.name})**")
                st.metric(f"📈 {RETURN_LABEL}", f"{user_choice['Return_Display']:.2f}%", help=help_return)
                st.metric("📉 CVaR Risk", f"{user_choice['CVaR_Risk_Display']:.2f}%", help=help_risk)
                st.metric("🏆 Sharpe Ratio", f"{user_choice['Prob_Sharpe']:.2f}", help=help_sharpe)
                st.metric("🌿 ESG Score", f"{user_choice['ESG_Score']:.1f}", help=help_esg)
                
                # ---  ADDING BACKTEST BUTTON FOR USER'S CHOICE ---
                if st.button(f"📊 Backtest Your Choice (Portfolio {user_choice.name})", key="backtest_user", use_container_width=True):
                    st.session_state.deep_link_period = period
                    st.session_state.deep_link_index = user_choice.name
                    st.session_state.deep_link_name = f"Your Choice (Portfolio {user_choice.name})"
                    st.session_state.deep_link_cash = initial_cash
                    st.switch_page("pages/5_📊_Backtester.py")
                # --- End of button ---

            with col_ai:
                st.markdown(f"**Our Recommendation (Portfolio {our_recommendation.name})**")
                st.metric(f"📈 {RETURN_LABEL}", f"{our_recommendation['Return_Display']:.2f}%", delta=f"{our_recommendation['Return_Display'] - user_choice['Return_Display']:.2f}%", help=help_return)
                st.metric("📉 CVaR Risk", f"{our_recommendation['CVaR_Risk_Display']:.2f}%", delta=f"{our_recommendation['CVaR_Risk_Display'] - user_choice['CVaR_Risk_Display']:.2f}%", delta_color="inverse", help=help_risk)
                st.metric("🏆 Sharpe Ratio", f"{our_recommendation['Prob_Sharpe']:.2f}", delta=f"{our_recommendation['Prob_Sharpe'] - user_choice['Prob_Sharpe']:.2f}", help=help_sharpe)
                st.metric("🌿 ESG Score", f"{our_recommendation['ESG_Score']:.1f}", delta=f"{our_recommendation['ESG_Score'] - user_choice['ESG_Score']:.1f}", help=help_esg)
                
                # ---  ADDING BACKTEST BUTTON FOR AI RECOMMENDATION ---
                if st.button(f"📊 Backtest AI Recommendation (Portfolio {our_recommendation.name})", key="backtest_ai", use_container_width=True, type="primary"):
                    st.session_state.deep_link_period = period
                    st.session_state.deep_link_index = our_recommendation.name
                    st.session_state.deep_link_name = f"AI Recommendation (Portfolio {our_recommendation.name})"
                    st.session_state.deep_link_cash = initial_cash
                    st.switch_page("pages/5_📊_Backtester.py")
                # --- End of button ---

            with st.expander("Show Recommended Portfolio Allocation"):
                rec_weights_series = weights.iloc[our_recommendation.name]
                rec_weights_df = pd.DataFrame(rec_weights_series[rec_weights_series > 0.01])
                rec_weights_df.columns = ['Weight']
                rec_weights_df = rec_weights_df.merge(sector_map, left_index=True, right_index=True)

                # --- ADDING CSV DOWNLOAD BUTTON FOR RECOMMENDED ALLOCATION ---
                csv_data_rec = convert_df_to_csv(rec_weights_df[['Weight', 'Sector']].sort_values(by='Weight', ascending=False))
                st.download_button(
                    label="📥 Download Recommended Allocation as CSV",
                    data=csv_data_rec,
                    file_name=f"Recommended_Portfolio_{period}_{our_recommendation.name}.csv",
                    mime='text/csv',
                    key="download_rec_csv" # Adding a unique key
                )
                

                col_pie_rec, col_tree_rec = st.columns(2)
                with col_pie_rec:
                    fig_pie_rec = px.pie(rec_weights_df, values='Weight', names=rec_weights_df.index, title="Rec. Allocation by Stock")
                    st.plotly_chart(fig_pie_rec, use_container_width=True, key="rec_pie_chart")
                with col_tree_rec:
                    fig_tree_rec = px.treemap(rec_weights_df, path=[px.Constant("Portfolio"), 'Sector', rec_weights_df.index], values='Weight', title="Rec. Allocation by Sector")
                    st.plotly_chart(fig_tree_rec, use_container_width=True, key="rec_tree_chart")

    st.divider()

    # =========================================
    # ---  PART 4: The 3D Explorer  ---
    # =========================================
    with st.expander("Show 3D Pareto Frontier (Advanced)"):
        st.header("Explore the Full 3D Pareto Frontier")
        st.markdown("Here you can see *all* optimal solutions found by the AI. This plot helps you understand the *inherent trade-offs* between all three goals.")
        
        fig_3d = px.scatter_3d(solutions,
                            x='CVaR_Risk_Display',
                            y='Return_Display',
                            z='ESG_Score',
                            color='Prob_Sharpe',
                            title=f'3D Pareto Frontier ({period})',
                            hover_data=['CVaR_Risk_Display', 'Return_Display', 'ESG_Score', 'Prob_Sharpe']
                            )
        
        fig_3d.update_layout(scene=dict(
                                xaxis_title='CVaR Risk (%)',
                                yaxis_title=f'{RETURN_LABEL} (%)',
                                zaxis_title='ESG Score'
                                ),
                                margin=dict(r=20, b=10, l=10, t=40))
        st.plotly_chart(fig_3d, use_container_width=True, key="3d_scatter_chart")

else:
    st.error(f"Failed to load data for period '{period}'.")
    st.page_link("Home.py", label="Go to Home Page", icon="🏠")
