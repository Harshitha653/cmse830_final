# app.py
import streamlit as st
import pandas as pd
from feature_engineering import preprocess_tweets, preprocess_news, preprocess_stocks, vectorize_text
from visualizations import (
    plot_stock_trends,
    plot_sentiment_distribution,
    plot_news_sentiment,
    plot_scatter_sentiment_vs_stock,
    plot_wordcloud
)
from utils import load_model, predict_sentiment

st.set_page_config(page_title="Market & Sentiment Dashboard", layout="wide")

# ------------------------------
# 1. LOAD DATA USING FILE UPLOADER
# ------------------------------
def load_data():
    st.sidebar.title("📁 Upload Your Datasets")

    stocks_file = st.sidebar.file_uploader("Upload Stock Dataset", type=["csv"])
    tweets_file = st.sidebar.file_uploader("Upload Tweets Dataset", type=["csv"])
    news_file   = st.sidebar.file_uploader("Upload News Dataset", type=["csv"])

    if not (stocks_file and tweets_file and news_file):
        st.warning("Please upload: **Stock CSV**, **Tweets CSV**, and **News CSV** to continue.")
        st.stop()

    # Automatically detect delimiter (comma/tab)
    df_stocks = pd.read_csv(stocks_file, sep=None, engine="python")
    df_tweets = pd.read_csv(tweets_file, sep=None, engine="python")
    df_news   = pd.read_csv(news_file,   sep=None, engine="python")

    # ----------------------------
    # Preprocess each dataset
    # ----------------------------
    df_stocks = preprocess_stocks(df_stocks)

    df_tweets = preprocess_tweets(df_tweets)
    df_tweets["sentiment_label"] = df_tweets["airline_sentiment"].map({
        "positive": 1,
        "neutral": 0,
        "negative": -1
    })

    df_news = preprocess_news(df_news)

    return df_stocks, df_tweets, df_news

df_stocks, df_tweets, df_news = load_data()

# ------------------------------
# 2. SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Stocks", "Tweets", "News", "Sentiment Prediction"])

# ------------------------------
# 3. OVERVIEW PAGE
# ------------------------------
if page == "Overview":
    st.title("📊 Market & Sentiment Dashboard")

    st.markdown("""
        This dashboard integrates **stock market data**, **tweets**, and **news headlines**
        to help analyze how public sentiment aligns with market trends.
    """)

    st.subheader("📁 Dataset Preview")
    st.write("### Stocks")
    st.dataframe(df_stocks.head())

    st.write("### Tweets")
    st.dataframe(df_tweets.head())

    st.write("### News")
    st.dataframe(df_news.head())

# ------------------------------
# 4. STOCKS PAGE
# ------------------------------
elif page == "Stocks":
    st.title("📈 Stock Analysis")

    ticker_list = df_stocks["ticker"].unique().tolist()
    selected_ticker = st.selectbox("Select Ticker", ticker_list)

    plot_stock_trends(df_stocks, selected_ticker)

    st.subheader("📊 Statistics")
    st.dataframe(df_stocks[df_stocks["ticker"] == selected_ticker].describe())

    st.subheader("📉 Returns & Volatility")
    df_selected = df_stocks[df_stocks["ticker"] == selected_ticker]
    st.line_chart(df_selected[["daily_return", "volatility"]])

# ------------------------------
# 5. TWEETS PAGE
# ------------------------------
elif page == "Tweets":
    st.title("💬 Tweets Sentiment Analysis")

    st.subheader("Sentiment Distribution")
    plot_sentiment_distribution(df_tweets)

    st.subheader("Word Cloud")
    plot_wordcloud(df_tweets, "text_clean")

    st.subheader("Sample Tweets")
    st.dataframe(df_tweets[["text", "airline_sentiment", "sentiment_label"]].head(20))

# ------------------------------
# 6. NEWS PAGE
# ------------------------------
elif page == "News":
    st.title("📰 News Sentiment Analysis")

    st.subheader("Sentiment Over Time")
    plot_news_sentiment(df_news)

    st.subheader("Word Cloud")
    plot_wordcloud(df_news, "headline_text_clean")

    st.subheader("Sample Headlines")
    st.dataframe(df_news.head(20))

# ------------------------------
# 7. SENTIMENT PREDICTION PAGE
# ------------------------------
elif page == "Sentiment Prediction":
    st.title("🔮 Predict Tweet Sentiment")

    tweet = st.text_area("Enter a tweet:")

    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model("ensemble_model.pkl")

    model = st.session_state.model_bundle["model"]
    vectorizer = st.session_state.model_bundle["vectorizer"]

    if st.button("Predict"):
        if tweet.strip() == "":
            st.warning("Please enter text.")
        else:
            pred = predict_sentiment(model, vectorizer, tweet)
            label = {1: "Positive", 0: "Neutral", -1: "Negative"}[pred]
            st.success(f"Sentiment: **{label}**")
