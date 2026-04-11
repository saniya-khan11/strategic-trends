import streamlit as st
import pandas as pd
import model2_var

# PAGE CONFIG
st.set_page_config(
    page_title="final_app",
    layout="wide",
    page_icon="📈"
)


# CUSTOM CSS 

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: #ffffff;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.big-font {
    font-size: 22px !important;
    font-weight: 600;
}

.small-text {
    color: #9ca3af;
}

</style>
""", unsafe_allow_html=True)
#card styling

st.markdown("""
<style>
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.card h4 {
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR

#st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
st.sidebar.title("StrategicTrends")
option = st.sidebar.radio(
    "Choose Section",
    [
        "Home",
        "Strategy 1: Transformer Picks",
        "Strategy 2: xgboost signals",
        "Strategy 3: Value at Risk",
    ]
)


# HOME PAGE

if option == "Home":
    # HEADER

    col1, col2 = st.columns([6,3 ])  # adjust ratio as needed

    with col1:
        st.title("StrategicTrends")
        st.caption("**Your Stock Search Ends Here!**")
        st.markdown("""
No charts.           
No technical analysis.             
Just **AI-driven decisions**.
""")

    with col2:
        st.image("logo.png", width=250)  # adjust width if needed

    #st.caption("AI-Powered Investing for Busy Professionals & Students")


    st.markdown("---")

    st.markdown("## Welcome to StrategicTrends")
    st.markdown("""
This platform simplifies investing using **AI-powered strategies** especially designed for working professionals and students who do not have enough time to do analysis on markets.

   """ )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <p class="big-font">🤖 AI Stock Strategy</p>
        <p class="small-text">
        Uses a Temporal Transformer model to identify high-potential stocks 
        for a 6-month horizon.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <p class="big-font">⚠️ Risk Management</p>
        <p class="small-text">
        Calculates Value at Risk (VaR) to estimate potential losses 
        in extreme market conditions.
        </p>
        </div>
        """, unsafe_allow_html=True)

   

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h4>🎯 Who is this for?</h4>
        <ul>
        <li>Professionals with no time for analysis</li>
        <li>Students learning investing</li>
        <li>Beginners entering markets</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Why StrategicTrends?</h4>
        <ul>
        <li>No charts, no confusion</li>
        <li>Pure AI-driven insights</li>
        <li>Simple decisions</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    

# STRATEGY 1

elif option == "Strategy 1: Transformer Picks":

    st.markdown("## 🤖 Transformer Model Strategy ")
    st.markdown("## Stock Recommendations to Invest for 6 months")

    st.markdown(""" 
<div class="card">
<b>Model:</b> Temporal Transformer <br>
<b>Horizon:</b> 6 Months <br>
<b>Goal:</b> Beat market using ranking-based selection
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")

    try:
        df = pd.read_csv("model1_output.csv")

        # Get Top 3 stocks
        top3 = df.head(3)

        stocks_html = "<br>".join(
        [
        f"{i+1}. <b>{row['Stock']}</b> (Score: {row['Score']:.4f})"
        for i, (_, row) in enumerate(top3.iterrows())
        ]
        )

        st.markdown(f"""
        <div class="card">
        <h4> Top 3 Stocks to Invest in for 6 months : </h4>
        {stocks_html}
        <br><br>
        
        </div>
        """, unsafe_allow_html=True)

    except:
        st.markdown("""
        <div class="card">
        ❌ No stock recommendations available<br>
        Run the model first
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Top Picks")

    try:
        df = pd.read_csv("model1_output.csv")
        top_df = df.head(10)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(top_df, use_container_width=True)

        with col2:
            st.markdown("### Score Visualization")
            st.bar_chart(top_df.set_index("Stock"))

        st.success("Model updated successfully")

    except:
        st.error("Run model1_tt.py to generate results")


# STRATEGY 2

elif option == "Strategy 2: xgboost signals":

    st.header("Xgboost Hybrid Multi-Factor Trading Model")
    st.caption("AI + Technical Indicators + Sentiment Analysis")

    st.markdown("""
This strategy combines:

- Technical Indicators (EMA, RSI, MACD)
- 🧠 Sentiment Analysis (News + FinBERT)
- Machine Learning (XGBoost)

Output: **Next-Day Buy/Sell/Hold Signals**
""")

    st.divider()

    try:
        # =========================
        # LOAD DATA
        # =========================
        df = pd.read_csv("model2_output.csv")
        st.success("✅ Latest Signals Loaded")

        # =========================
        # USER INPUT ONLY
        # =========================
        search = st.text_input("🔎 Enter Stock Name (e.g., RELIANCE, TCS)")

        st.markdown("### Horizon")
        st.info("Next Trading Day")

        st.divider()

        # =========================
        # SHOW RESULT ONLY IF USER ENTERS
        # =========================
        if search:

            filtered = df[df["Stock"].str.upper() == search.upper()]

            if not filtered.empty:

                result = filtered.iloc[0]
                signal = result["Next Day Signal"]

                # SIGNAL STYLE
                if signal.lower() == "buy":
                    color = "green"
                    emoji = "🟢"
                elif signal.lower() == "sell":
                    color = "red"
                    emoji = "🔴"
                else:
                    color = "orange"
                    emoji = "🟡"

                # METRICS
                col1, col2, col3 = st.columns(3)

                col1.metric("Price", f"₹{result['Current Price']}")
                col2.metric("Signal", f"{emoji} {signal}")
                col3.metric("Confidence", f"{result['Confidence (%)']}%")

                st.divider()

                # =========================
                # INSIGHT
                # =========================
                st.subheader("Model Insight")

                if signal.lower() == "buy":
                    st.success("Strong bullish signals detected across momentum, trend, and sentiment.")
                elif signal.lower() == "sell":
                    st.error("Bearish indicators dominate. Risk of downside movement.")
                else:
                    st.warning("Mixed signals. Market indecisive.")

            else:
                st.error("❌ Stock not found. Please enter a valid stock name.")

        else:
            st.info(" Enter a stock name to see AI prediction.")

    except Exception as e:
        st.error("❌ model4_output.csv not found. Run model first.")
        st.code("python model4_xgboost.py")
# ======================================================
# STRATEGY 3
# ======================================================
elif option == "Strategy 3: Value at Risk":

    st.markdown("## ⚠️ Value at Risk Calculation for your Portfolio")

    st.markdown("""
<div class="card">
<b>What is Value at Risk(VaR)?</b><br>
It gives you the a single - day maximum expected loss of your portfolio with 95 % confidence.
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## Know your portfolio daily expected loss!")
    st.markdown("### Select Stocks of your Portfolio:")

    stock_options = list(model2_var.NIFTY50_MAP.keys())

    selected_stocks = st.multiselect(
        "Choose Stocks",
        stock_options
    )

    st.markdown("")

    if st.button("🚀 Run Risk Analysis", use_container_width=True):

        if not selected_stocks:
            st.warning("Select at least one stock")
        else:
            with st.spinner("Analyzing portfolio risk..."):

                results = model2_var.calculate_var(selected_stocks)

            if "error" in results:
                st.error(results["error"])
            else:
                st.success("Analysis Complete")

                st.markdown("### 📊 Portfolio Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric("VaR", results["VaR"]*100,"%")
                col2.metric("Mean Return", results["Mean Return"]*100,"%")
                col3.metric("Volatility", results["Volatility"]*100,"%")

                st.markdown("---")

                st.markdown("### 🧠 Interpretation")

                risk_level = "Low Risk ✅" if results["VaR"] < 0.02 else "High Risk ⚠️"

                st.markdown(f"""
<div class="card">
<b>Risk Level:</b> {risk_level} <br><br>

There is a <b>5% chance</b> your portfolio may lose more than <b>{results["VaR"]*100:.2f}% in a single day</b>.
</div>
""", unsafe_allow_html=True)

                st.markdown("### Selected Stocks")
                st.write(results["Selected Stocks"])

