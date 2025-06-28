# 📊 Trader Behavior Insights & Profitability Prediction App

This Streamlit web app analyzes the relationship between market sentiment and trader behavior using real-world trading and sentiment datasets. It offers powerful visualizations and a predictive model to estimate whether given trading conditions are likely to result in a **profit ("Win") or a loss ("Loss")**.

## 🔍 Key Features

### 1. **Insights Dashboard**
- **Trader Behavior Clustering**: Groups traders into behavior-based clusters using KMeans based on their trading volume, fees, profitability, and consistency.
- **Market Sentiment Correlation**: Shows how the Fear & Greed Index impacts metrics like total trades, average size, win ratio, etc.
- **Market Sentiment vs Total PnL**: Compares market sentiment with profit/loss trends to reveal parallel movements and potential predictive signals.
- **Lagged Correlation Analysis**: Evaluates whether previous days' market sentiment can influence today’s trading results.
- **Rolling PnL & Volatility**: Tracks profitability trends and associated volatility over time.

### 2. **Predict PnL Class**
- Input values for market sentiment, number of trades, trade size, fees, and win ratio.
- Get a prediction: **Will it be a Win or Loss?** – using a trained machine learning model (`RandomForestClassifier`).

---

## 🧠 Technologies Used
- **Streamlit**: For the interactive dashboard and web interface.
- **scikit-learn**: Machine learning models and preprocessing.
- **Pandas** & **NumPy**: Data manipulation.
- **Matplotlib** & **Seaborn**: Data visualization.
- **Joblib**: Model serialization and loading.

---
🌐 Live Demo
Access the deployed app here:
🔗 https://rishitaasharmaa-trader-behavior-insights-app-g2if1j.streamlit.app/


