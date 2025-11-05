# 🚀 AI-Driven Multi-Objective ESG Portfolio Optimizer

This is a full-stack, end-to-end Term Project for a 6-credit B.S. Data Science and AI course. It is a web application that builds optimal investment portfolios by balancing three conflicting objectives: **Profitability**, **Risk**, and **ESG (Environmental, Social, Governance) Score**.

The application uses **XGBoost** for short-term profit predictions and a **Multi-Objective Evolutionary Algorithm (NSGA-II)** to discover the 3D "Pareto Optimal Frontier" for **9 distinct time horizons**.

---

##  Core Features

* **Dynamic Time Horizons:** Creates 9 distinct portfolio models, separated into:
    * **Short-Term Tactical (AI-Driven):** 3-Day, 7-Day, 15-Day, 30-Day, and 3-Month models using XGBoost to predict profit probability.
    * **Long-Term Strategic (Historical):** 6-Month, 1-Year, 3-Year, and 5-Year models based on historical annualized returns.
* **Multi-Objective Optimization:** The core of the project. It finds 100+ optimal portfolios that represent the best possible trade-offs between:
    1.  **Return (Profit):** AI-predicted profit probability (XGBoost) or historical annualized returns.
    2.  **Risk (CVaR):** Conditional Value at Risk, a state-of-the-art metric that measures "tail risk" (the risk of extreme losses).
    3.  **ESG Score:** A 0-100 score based on a company's environmental, social, and governance ratings.
* **Full-Stack Web App:** A dynamic 5-page Streamlit application that allows users to:
    * Select their desired time horizon (e.g., "3D Sprint" or "5Y Vision").
    * Interactively filter the "Pareto Frontier" to find a portfolio that matches their personal goals.
    * Receive an AI-driven recommendation for a "better" portfolio.
* **Explainable AI (XAI):** A "Stock Deep Dive" page that shows *why* the AI is making its predictions, displaying historical profit probabilities and feature importances for any stock.
* **One-Click Backtesting:** A fully integrated 5-year backtester that allows any portfolio (user-selected or AI-recommended) to be tested against the NIFTY 50 benchmark.

---

## 🛠️ Tech Stack & Methodology

### 1. Backend (`main.py`)
The backend is a standalone Python script that performs all heavy data engineering and AI modeling.
* **Data:** `yfinance` (Price Data), `pandas` (Cleaning & Merging), `TA-Lib` (Feature Engineering).
* **AI Model:** `XGBoost` (Gradient Boosting) is used to train 5 separate `XGBClassifier` models to predict profit *probability* for 3, 7, 15, 30, and 90-day horizons.
* **Risk Model:** `Conditional Value at Risk (CVaR)` is calculated from 5 years of historical returns.
* **Optimizer:** `pymoo` is used to run the **NSGA-II** evolutionary algorithm, finding the 3D Pareto front for all 9 time horizons.
* **Output:** The script saves 18 `.csv` files (9 for weights, 9 for solutions) plus a `master_data_for_app.csv` file, which the frontend app consumes.

### 2. Frontend (`Home.py` & `pages/`)
The frontend is a multi-page Streamlit application.
* **UI:** `Streamlit`, `plotly` (for interactive 3D scatter plots, pie charts, and treemaps).
* **State Management:** `st.session_state` is used to create a seamless user experience, passing portfolio choices and recommendations between pages for one-click backtesting.

---

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ExcelsiorSR/Project_AI_ESGPO.git](https://github.com/ExcelsiorSR/Project_AI_ESGPO.git)
    cd Project_AI_ESGPO
    ```

2.  **Setup Environment:**
    ```bash
    # Create and activate a conda environment
    conda create -n ESGPO python=3.10
    conda activate ESGPO

    # Install all required packages
    pip install -r requirements.txt
    ```
    *Note: `TA-Lib` can be difficult to install. If the pip install fails, you may need to install it via Conda (`conda install -c conda-forge ta-lib`) or by downloading the wheel file for your system.*

3.  **Run the AI Engine (One-Time Only):**
    This script will take ~1-2 minutes to run. It downloads 5 years of data, runs the 9 optimization models, and saves all the `.csv` files.
    ```bash
    python main.py
    ```

4.  **Launch the Web App:**
    This runs the Streamlit server.
    ```bash
    streamlit run Home.py
    ```

---

## 📁 Project Structure

This project is organized into two main parts: a data processing "engine" (`main.py`) and a multi-page Streamlit web application (the `Home.py` and `pages/` directory).
* `Project_ES GPO/` (Root Folder)
    * `pages/` (Holds all the app's sub-pages)
        * `1_🚀_Short_Term_Optimizer.py`
        * `2_⏳_Long_Term_Optimizer.py`
        * `3_🔬_Portfolio_Builder.py`
        * `4_🕯️_Stock_Deep_Dive.py`
        * `5_📊_Backtester.py`
    * `Home.py` (The main "Welcome" landing page)
    * `main.py` (The "engine" script, run once to process data)
    * `requirements.txt` (List of all Python libraries to install)
    * `final_data.csv` (The raw NIFTY 50 ESG data - Input)
    * `master_data_for_app.csv` (Generated by main.py)
    * `backtest_price_data.csv` (Generated by main.py)
    * `optimized_solutions_3D.csv` (Generated by main.py)
    * `optimized_weights_3D.csv` (Generated by main.py)
    * `...` (and all other generated result files)
      
### Key Components

* **`main.py` (The "Engine")**
    This is the core data processing script. It must be run **once** before launching the app. It reads the raw `final_data.csv`, performs all the complex optimization calculations, and generates all the necessary result files (like `optimized_weights_3D.csv`, `backtest_price_data.csv`, etc.).

* **`Home.py` & `pages/` (The "App")**
    This is the Streamlit web application.
    * `Home.py` is the main welcome page.
    * The scripts in the `pages/` directory create the different tabs you see in the app's sidebar (e.g., "Short Term Optimizer," "Backtester").
    * These app files **read** the result files generated by `main.py` to display the charts, tables, and analysis.

* **Data Files**
    * **Input:** `final_data.csv` is the only manual input file required by the engine.
    * **Output:** All other `.csv` files are **generated** by `main.py`. The app relies on these files to function.

## 🛠️ Installation

Before running the project, you need to install all the required Python libraries.

1.  Open your terminal or command prompt.
2.  Navigate to the project's root folder (`Project_ES GPO/`).
3.  Run the following command to install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

There are two main steps to run this project:

### Step 1: Run the Data Engine (Once)

You must first run the `main.py` script to perform all the calculations and generate the result files that the app needs.

```bash
python main.py
```
(This may take a few minutes to complete. You only need to do this once, or whenever the final_data.csv changes.)

### Step 2: Launch the Streamlit App
Once the engine has finished, you can launch the interactive web application.
```bash
streamlit run Home.py
```
This will open the application in your default web browser.
## ✨ Features / App Overview

This application provides a multi-faceted toolset for analyzing and optimizing ESG (Environmental, Social, and Governance) portfolios based on NIFTY 50 data.

Here is a breakdown of each module:

* **🏠 Home:**
    *This is where you'll land at first. It includes all the major page navigations you can have i.e. from direct links to optimizer to the Backtester as well. It includes the redirection link to the detailed Do's and Don'ts guidelines from SEBI. *

* **🚀 Short Term Optimizer:**
    *Here you'll find links to all the major Short Term Portfolio statistics i.e. from 3 days to 90 days.*

* **⏳ Long Term Optimizer:**
    *Here you'll find links to all the major Short Term Portfolio statistics i.e. from 6 months to 60 months.*

* **🔬 Portfolio Builder:**
    *This is where the real magic happens. Here you'll be able to see the detailed portfolio allocations you need to do in order to get the profit probability predicted in the page. You can as well customise your own portfolio by changing the Profit Goals, Risk Tolerance and Ethical Priority and can as well find a better portfolio than your selected one and can directly backtest it from here.*

* **🕯️ Stock Deep Dive:**
    *This page provides a detailed, granular analysis of a single selected stock, showing its historical performance, risk metrics, and individual ESG score breakdown. Along with it you can also find the detailed metrics the model studied for giving the result.*

* **📊 Backtester:**
    *(This page allows users to test the historical performance of the optimized portfolios (both short and long-term) to see how they would have performed in the past provided they had invested in the stocks suggested.*

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
