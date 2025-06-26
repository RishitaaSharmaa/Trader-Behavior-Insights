import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Trader Insights Dashboard", layout="wide")

merged_df = pd.read_csv("merged_data.csv")
model = joblib.load("model.pkl")

page = st.sidebar.selectbox("Select Page", ["📈 Insights Dashboard", "🧮 Predict PnL Class"])

if page == "📈 Insights Dashboard":
    st.title("📈 Trader Insights & Strategy Guide")
    st.markdown("Get a deeper understanding of market behavior, sentiment impact, and how traders perform across different conditions.")

    st.subheader("🧠 Trader Behavior Clusters (KMeans)")
    features_for_clustering = merged_df[[
        'total_trades', 'total_pnl', 'avg_trade_size', 'total_fees', 'win_ratio']].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)

    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_df['cluster'] = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    merged_df['pca1'] = pca_features[:, 0]
    merged_df['pca2'] = pca_features[:, 1]

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.scatterplot(data=merged_df, x='pca1', y='pca2', hue='cluster', palette='Set2', ax=ax1)
    ax1.set_title("KMeans Clusters of Trader Behavior")
    st.pyplot(fig1)
    st.markdown("Each cluster represents a group of traders with similar behavior, based on their activity, profitability, trade size, and win rate.")
    st.info("👉 Conservative traders tend to fall in one cluster, focusing on consistent small wins. Aggressive traders might be in another, showing high PnL variability. Identify your cluster to optimize strategies.")

    st.subheader("📊 Correlation: Market Sentiment vs Trading Metrics")
    corr = merged_df[["value", "total_trades", "total_pnl", "avg_trade_size", "total_fees", "win_ratio"]].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Between Market Sentiment and Trader Metrics")
    st.pyplot(fig2)
    st.markdown("This matrix shows how the Fear and Greed Index (value) influences trading behavior.")
    st.info("👉 Traders tend to trade more and incur higher fees when the market shows 'Greed'. However, profitability (PnL) doesn't always rise in tandem — be cautious of irrational exuberance.")

    st.subheader("📈 Sentiment vs Total PnL Over Time")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(pd.to_datetime(merged_df['date']), merged_df['value'], label='Fear & Greed Index', color='red')
    ax3.plot(pd.to_datetime(merged_df['date']), merged_df['total_pnl'], label='Total PnL', color='blue')
    ax3.legend()
    ax3.set_title("Market Sentiment vs Trader Total PnL Over Time")
    st.pyplot(fig3)
    st.markdown("The graph reveals that PnL often spikes during sentiment extremes, suggesting market inefficiencies or overreactions.")
    st.info("👉 Look for periods of extreme fear or greed to exploit volatility, but use proper risk controls. Large swings may offer breakout or reversal opportunities.")

    st.subheader("📦 PnL Distribution by Market Sentiment")
    fig4, ax4 = plt.subplots(figsize=(6, 2))
    sns.boxplot(data=merged_df, x='classification', y='total_pnl', ax=ax4)
    ax4.set_title("Total Daily PnL by Sentiment Classification")
    st.pyplot(fig4)
    st.markdown("This boxplot shows how profit and loss varies under different market sentiment labels: Fear, Greed, Neutral, etc.")
    st.info("👉 While average returns may be modest, extreme values appear frequently in both 'Extreme Fear' and 'Greed'. Traders can benefit from adopting momentum or contrarian tactics during these times.")

    st.subheader("⏳ Lagged Correlation Between Sentiment and Trading Metrics")
    for lag in range(1, 4):
        merged_df[f'value_lag{lag}'] = merged_df['value'].shift(lag)

    lagged_corr = merged_df[[
        'value', 'value_lag1', 'value_lag2', 'value_lag3',
        'total_trades', 'total_pnl', 'avg_trade_size', 'total_fees', 'win_ratio']].dropna().corr()

    fig5, ax5 = plt.subplots(figsize=(7, 3))
    sns.heatmap(lagged_corr, annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Lagged Sentiment Correlation with Trader Metrics")
    st.pyplot(fig5)
    st.markdown("This heatmap reveals how past sentiment (1 to 3 days ago) correlates with trading behavior today.")
    st.info("👉 Strong correlation between value_lag1 and PnL may suggest yesterday's sentiment affects today's trading outcome — useful for forecasting setups.")

    st.subheader("📉 Rolling PnL & Market Volatility")
    merged_df['rolling_pnl'] = merged_df['total_pnl'].rolling(window=5).mean()
    merged_df['volatility'] = merged_df['total_pnl'].rolling(window=5).std()
    fig6, ax6 = plt.subplots(figsize=(7, 3))
    ax6.plot(pd.to_datetime(merged_df['date']), merged_df['rolling_pnl'], label='5-Day Rolling PnL')
    ax6.plot(pd.to_datetime(merged_df['date']), merged_df['volatility'], label='Volatility')
    ax6.legend()
    ax6.set_title("5-Day Rolling PnL and Volatility")
    st.pyplot(fig6)
    st.markdown("Tracks 5-day average profits and risk level. Useful for identifying smooth uptrends or turbulent markets.")
    st.info("👉 When volatility increases and rolling PnL improves — that's a bullish condition for active traders. If volatility rises but PnL doesn't — scale back.")

    st.markdown("---")
    st.caption("Insights derived from historical trading and sentiment analysis data to guide smart trading behavior.")

elif page == "🧮 Predict PnL Class":
    st.title("🧮 Predict Trader Profitability Class")
    st.markdown("Enter market and trading conditions to predict the profit/loss classification.")

    with st.form("pnl_form"):
        value = st.number_input("📉 Market Sentiment Value (Fear-Greed Index)", min_value=0.0, max_value=100.0, value=50.0)
        total_trades = st.number_input("🔢 Total number of trades", min_value=0)
        avg_trade_size = st.number_input("📏 Average size of each trade")
        total_fees = st.number_input("💸 Total transaction fees")
        win_ratio = st.slider("🏆 Win Ratio (0 to 1)", min_value=0.0, max_value=1.0, step=0.01)
        submit = st.form_submit_button("Predict PnL Class")

    if submit:
        input_df = pd.DataFrame.from_dict({
            'value': [value],
            'total_trades': [total_trades],
            'avg_trade_size': [avg_trade_size],
            'total_fees': [total_fees],
            'win_ratio': [win_ratio]
        })

        prediction = model.predict(input_df)[0]
        result = "Win" if prediction == 1 else "Loss"
        st.success(f"📊 Predicted Outcome: **{result}**")