import streamlit as st
import pandas as pd
from feature_engineering import preprocess_tweets, preprocess_news, preprocess_stocks
from visualizations import (
    plot_stock_trends,
    plot_sentiment_distribution,
    plot_news_sentiment,
    plot_wordcloud
)
from utils import load_model, predict_sentiment

st.set_page_config(page_title="Market & Sentiment Dashboard", layout="wide")

# -------------------------------------------------------
# 1. LOAD DATA FROM GOOGLE DRIVE
# -------------------------------------------------------
@st.cache_data
def load_data():
    # ★ Direct download links (REPLACE WITH YOUR FINAL LINKS)
    stocks_url = "https://drive.google.com/uc?export=download&id=15OmtMwBNGlE_UoB0vVnSyWhl4SBRDuEz"
    tweets_url = "https://drive.google.com/uc?export=download&id=1BQuO-3015cvVniWZcKXzYos5PlpkqtPn"
    news_url   = "https://drive.google.com/uc?export=download&id=1F4ffWPMfH1k2Oa17_lUZouNlI4GKXyvm"

    # Auto-detect delimiter: comma or tab
    df_stocks = pd.read_csv(stocks_url, sep=None, engine="python")
    df_tweets = pd.read_csv(tweets_url, sep=None, engine="python")
    df_news   = pd.read_csv(news_url,   sep=None, engine="python")

    # Preprocess each dataset
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

# -------------------------------------------------------
# 2. SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Stocks", "Tweets", "News", "Sentiment Prediction"])

# -------------------------------------------------------
# 3. OVERVIEW PAGE
# -------------------------------------------------------
if page == "Overview":
    st.title("📊 Market & Sentiment Dashboard")
    st.write("Integrated view of stock prices, tweets, and news sentiment.")

    st.subheader("Stock Sample")
    st.dataframe(df_stocks.head())

    st.subheader("Tweets Sample")
    st.dataframe(df_tweets.head())

    st.subheader("News Sample")
    st.dataframe(df_news.head())

# -------------------------------------------------------
# 4. STOCKS PAGE
# -------------------------------------------------------
elif page == "Stocks":
    st.title("📈 Stock Analysis")

    ticker_list = df_stocks["ticker"].unique().tolist()
    selected_ticker = st.selectbox("Select Ticker", ticker_list)

    plot_stock_trends(df_stocks, selected_ticker)

    st.subheader("Statistics")
    st.dataframe(df_stocks[df_stocks["ticker"] == selected_ticker].describe())

    st.subheader("Daily Return & Volatility")
    df_sel = df_stocks[df_stocks["ticker"] == selected_ticker]
    st.line_chart(df_sel[["daily_return", "volatility"]])

# -------------------------------------------------------
# 5. TWEETS PAGE
# -------------------------------------------------------
elif page == "Tweets":
    st.title("💬 Tweets Sentiment Analysis")

    st.subheader("Sentiment Distribution")
    fig = plot_sentiment_distribution(df_tweets)
    st.plotly_chart(fig)

    st.subheader("Word Cloud")
    plot_wordcloud(df_tweets, "text_clean")

    st.subheader("Sample Tweets")
    st.dataframe(df_tweets[["text", "airline_sentiment", "sentiment_label"]].head(20))

# -------------------------------------------------------
# 6. NEWS PAGE
# -------------------------------------------------------
elif page == "News":
    st.title("📰 News Sentiment Analysis")

    st.subheader("News Count Over Time")
    plot_news_sentiment(df_news)

    st.subheader("News Word Cloud")
    plot_wordcloud(df_news, "headline_text_clean")

    st.subheader("Sample Headlines")
    st.dataframe(df_news.head(20))

# -------------------------------------------------------
# 7. SENTIMENT PREDICTION
# -------------------------------------------------------
elif page == "Sentiment Prediction":
    st.title("🔮 Predict Tweet Sentiment")

    text = st.text_area("Enter Tweet:")

    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model("ensemble_model.pkl")

    bundle = st.session_state.model_bundle
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("Enter a tweet.")
        else:
            pred = predict_sentiment(bundle, vectorizer, text)
            label = {1: "Positive", 0: "Neutral", -1: "Negative"}[pred]
            st.success(f"Predicted Sentiment: **{label}**")
