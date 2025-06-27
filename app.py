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

page = st.sidebar.selectbox("Select Page", ["Insights Dashboard", "Predict PnL Class"])

if page == "Insights Dashboard":
    st.title("Trader Insights & Strategy Guide")
    st.markdown("Gain a comprehensive understanding of trading behavior, market sentiment impact, and strategies used by traders under varying conditions.")

    st.subheader("Trader Behavior Clusters (KMeans Clustering)")
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
    st.markdown("Each cluster represents a group of traders with similar behavior based on key trading metrics.")
    st.info("Conservative traders generally form one cluster, often emphasizing risk control and consistency. Aggressive traders typically form a separate cluster, characterized by higher variability in PnL. This segmentation helps tailor strategies according to trader type.")

    st.subheader("Correlation Between Market Sentiment and Trader Metrics")
    corr = merged_df[["value", "total_trades", "total_pnl", "avg_trade_size", "total_fees", "win_ratio"]].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Between Market Sentiment and Trader Metrics")
    st.pyplot(fig2)
    st.markdown("This heatmap provides an overview of how sentiment data (Fear and Greed Index) correlates with trading statistics.")
    st.info("For example, higher greed scores are often associated with more trades and fees, but not always with higher profits. A weak correlation between sentiment and PnL may indicate that emotions drive volume but not necessarily outcomes. Look for variables with strong correlations to derive actionable insights.")

    
    st.subheader("Market Sentiment vs Total PnL")
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(pd.to_datetime(merged_df['date']), merged_df['value_z'], label='Fear & Greed Index', color='red')
    ax4.plot(pd.to_datetime(merged_df['date']), merged_df['total_pnl_z'], label='Total PnL', color='blue')
    ax4.legend()
    ax4.set_title("Market Sentiment vs Trader Total PnL Over Time")
    st.pyplot(fig4)
    st.markdown("This visualization shows how trader profitability and market sentiment behave over time when both are scaled comparably.")
    st.info("Notice how significant spikes or drops in sentiment often precede or coincide with large changes in PnL. This alignment helps in spotting potential predictive relationships. Sharp sentiment shifts can act as early warning signals for volatility or trading opportunities.")

    st.subheader("Lagged Correlation Between Sentiment and Trading Metrics")
    for lag in range(1, 4):
        merged_df[f'value_lag{lag}'] = merged_df['value'].shift(lag)

    lagged_corr = merged_df[[
        'value', 'value_lag1', 'value_lag2', 'value_lag3',
        'total_trades', 'total_pnl', 'avg_trade_size', 'total_fees', 'win_ratio']].dropna().corr()

    fig5, ax5 = plt.subplots(figsize=(7, 3))
    sns.heatmap(lagged_corr, annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title("Lagged Sentiment Correlation with Trader Metrics")
    st.pyplot(fig5)
    st.markdown("This analysis explores whether previous days' sentiment scores influence current trading activity.")
    st.info("A strong correlation between value_lag1 and total_pnl indicates that yesterday's market sentiment has a measurable influence on today's trader performance. These findings can enhance model-based forecasting strategies.")

    st.subheader("5-Day Rolling Average and Volatility of Total PnL")
    merged_df['rolling_pnl'] = merged_df['total_pnl'].rolling(window=5).mean()
    merged_df['volatility'] = merged_df['total_pnl'].rolling(window=5).std()
    fig6, ax6 = plt.subplots(figsize=(7, 3))
    ax6.plot(pd.to_datetime(merged_df['date']), merged_df['rolling_pnl'], label='5-Day Rolling PnL')
    ax6.plot(pd.to_datetime(merged_df['date']), merged_df['volatility'], label='Volatility')
    ax6.legend()
    ax6.set_title("5-Day Rolling PnL and Volatility")
    st.pyplot(fig6)
    st.markdown("This graph visualizes both the smoothed profitability and market risk across time, offering a view of stability vs unpredictability in trader returns.")
    st.info("High volatility with consistent gains may indicate strong breakout trends, while high volatility with declining returns could signal potential reversals or instability. This metric helps manage risk exposure.")

    st.markdown("---")
    st.caption("Insights are based on historic trading and sentiment data to help you refine strategies and anticipate market behavior.")

elif page == "Predict PnL Class":
    st.title("Predict Trader Profitability Class")
    st.markdown("Provide market sentiment and trading condition inputs below to classify whether the resulting trading outcome would be profitable or not.")

    with st.form("pnl_form"):
        value = st.number_input("Market Sentiment Value (Fear-Greed Index)", min_value=0.0, max_value=100.0, value=50.0)
        total_trades = st.number_input("Total number of trades", min_value=0)
        avg_trade_size = st.number_input("Average size of each trade")
        total_fees = st.number_input("Total transaction fees")
        win_ratio = st.slider("Win Ratio (between 0 and 1)", min_value=0.0, max_value=1.0, step=0.01)
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
        st.success(f"Predicted Outcome: {result}")
