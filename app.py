import streamlit as st
import pandas as pd

from feature_engineering import (
    preprocess_tweets,
    preprocess_news,
    preprocess_stocks,
)
from visualizations import (
    plot_stock_trends,
    plot_sentiment_distribution,
    plot_news_sentiment,
    plot_wordcloud,
)
from utils import load_model, predict_sentiment

st.set_page_config(page_title="Market & Sentiment Dashboard", layout="wide")


# -------------------------------------------------------
# 1. LOAD DATA FROM GOOGLE DRIVE
# -------------------------------------------------------
@st.cache_data
def load_data():
    # ✅ Direct-download Google Drive links (replace IDs if you change files)
    stocks_url = "https://drive.google.com/uc?export=download&id=15OmtMwBNGlE_UoB0vVnSyWhl4SBRDuEz"
    tweets_url = "https://drive.google.com/uc?export=download&id=1BQuO-3015cvVniWZcKXzYos5PlpkqtPn"
    news_url   = "https://drive.google.com/uc?export=download&id=1F4ffWPMfH1k2Oa17_lUZouNlI4GKXyvm"

    # Auto-detect delimiter (comma / tab)
    df_stocks = pd.read_csv(stocks_url, sep=None, engine="python")
    df_tweets = pd.read_csv(tweets_url, sep=None, engine="python")
    df_news   = pd.read_csv(news_url,   sep=None, engine="python")

    # Preprocess each dataset
    df_stocks = preprocess_stocks(df_stocks)
    df_tweets = preprocess_tweets(df_tweets)
    df_news   = preprocess_news(df_news)

    # Map text sentiment → numeric label for downstream plots
    if "airline_sentiment" in df_tweets.columns:
        df_tweets["sentiment_label"] = df_tweets["airline_sentiment"].map(
            {"positive": 1, "neutral": 0, "negative": -1}
        )

    return df_stocks, df_tweets, df_news


df_stocks, df_tweets, df_news = load_data()


# -------------------------------------------------------
# 2. SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Stocks", "Tweets", "News", "Sentiment Prediction", "Modeling Details"],
)


# -------------------------------------------------------
# 3. OVERVIEW PAGE
# -------------------------------------------------------
if page == "Overview":
    st.title("📊 Market & Sentiment Dashboard")

    st.markdown(
        """
This dashboard brings together **three signals**:

- Daily **stock prices** for multiple tickers  
- **Tweets** with labeled sentiment  
- Short **news headlines** with publication dates  

The goal is to explore how **public sentiment** (from Twitter and news) relates to
market behavior over time.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stocks sample")
        st.dataframe(df_stocks.head())
    with col2:
        st.subheader("Tweets sample")
        st.dataframe(df_tweets.head())

    st.subheader("News sample")
    st.dataframe(df_news.head())


# -------------------------------------------------------
# 4. STOCKS PAGE
# -------------------------------------------------------
elif page == "Stocks":
    st.title("📈 Stock Analysis")

    if "ticker" not in df_stocks.columns:
        st.error("Expected 'ticker' column in stocks dataset.")
    else:
        ticker_list = sorted(df_stocks["ticker"].unique().tolist())
        selected_ticker = st.selectbox("Select ticker", ticker_list)

        st.subheader(f"Price trend for {selected_ticker}")
        plot_stock_trends(df_stocks, selected_ticker)

        st.subheader("Summary statistics")
        df_sel = df_stocks[df_stocks["ticker"] == selected_ticker]
        st.dataframe(df_sel[["open", "high", "low", "close", "volume", "daily_return", "volatility"]].describe())

        st.subheader("Daily return & volatility over time")
        st.line_chart(
            df_sel.set_index("date")[["daily_return", "volatility"]]
        )


# -------------------------------------------------------
# 5. TWEETS PAGE
# -------------------------------------------------------
elif page == "Tweets":
    st.title("💬 Tweets Sentiment Analysis")

    st.subheader("Sentiment distribution")
    fig = plot_sentiment_distribution(df_tweets)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tweet word cloud")
    plot_wordcloud(df_tweets, "text_clean")

    st.subheader("Sample labeled tweets")
    cols_to_show = [c for c in ["text", "airline_sentiment", "sentiment_label"] if c in df_tweets.columns]
    st.dataframe(df_tweets[cols_to_show].head(20))


# -------------------------------------------------------
# 6. NEWS PAGE
# -------------------------------------------------------
elif page == "News":
    st.title("📰 News Sentiment / Coverage")

    st.subheader("News volume over time")
    plot_news_sentiment(df_news)

    st.subheader("News headline word cloud")
    plot_wordcloud(df_news, "headline_text_clean")

    st.subheader("Sample headlines")
    st.dataframe(df_news[["publish_date", "headline_text", "headline_text_clean"]].head(20))


# -------------------------------------------------------
# 7. SENTIMENT PREDICTION PAGE
# -------------------------------------------------------
elif page == "Sentiment Prediction":
    st.title("🔮 Tweet Sentiment Prediction")

    st.markdown(
        """
This page uses a pre-trained **ensemble classifier** (Random Forest + Naive Bayes)
trained on labeled tweets.  
The model is stored in `ensemble_model.pkl` and loaded at runtime.
"""
    )

    text = st.text_area("Enter a tweet:")

    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model("ensemble_model.pkl")

    bundle = st.session_state.model_bundle
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    if st.button("Predict sentiment"):
        if text.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            pred = predict_sentiment(bundle, vectorizer, text)
            label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
            st.success(f"Predicted sentiment: **{label_map.get(pred, 'Unknown')}**")


# -------------------------------------------------------
# 8. MODELING DETAILS PAGE
# -------------------------------------------------------
elif page == "Modeling Details":
    st.title("🧠 Modeling & Data Pipeline Details")

    st.markdown(
        """
### 1. Datasets

**a) Stock prices**

- Source: daily OHLCV data for multiple tickers  
- Columns used:
  - `Date` → parsed to `date` (datetime)
  - `Open`, `High`, `Low`, `Close`, `Volume`, `OpenInt`
  - `Ticker` (string identifier for each stock)

From this we engineer:

- `daily_return` = percentage change of `close`
- `volatility` = 7-day rolling standard deviation of `daily_return`  

---

**b) Tweets dataset (airline sentiment)**

- Source: labeled airline tweet sentiment dataset  
- Core columns:
  - `text` – raw tweet text
  - `airline_sentiment` – label in {negative, neutral, positive}
  - `tweet_created` – timestamp of tweet

In the preprocessing step, we:

- lower-case the text  
- remove URLs, mentions, punctuation  
- remove English stopwords  
- store cleaned text in `text_clean`  
- map sentiment to numeric labels:
  - negative → -1
  - neutral → 0
  - positive → 1

---

**c) News headlines dataset**

- Columns:
  - `headline_text` – short news headline
  - `publish_date` – publication date (YYYYMMDD)

Preprocessing:

- convert `publish_date` to datetime
- clean `headline_text` similarly to tweets
- store the result in `headline_text_clean`

---

### 2. Text preprocessing (shared logic)

For tweets and news, we apply the same `clean_text` function:

- convert to lowercase  
- remove URLs (`http`, `https`, `www`)  
- strip mentions and simple hashtags  
- remove punctuation  
- remove English stopwords using scikit-learn's built-in list (`ENGLISH_STOP_WORDS`)  
- collapse multiple spaces  

This gives us compact, model-ready features in `text_clean` and `headline_text_clean`.

---

### 3. Sentiment model training (offline)

The sentiment classifier is trained in a separate script (`model.py`) on the tweets dataset:

1. Load the labeled tweets  
2. Apply regex-based cleaning to build `clean_text`  
3. Split into train/test with `train_test_split` (stratified)  
4. Vectorize text using **TF-IDF**:
   - up to 5,000 features  
   - unigrams + bigrams (`ngram_range=(1, 2)`)  
5. Train two models:
   - `RandomForestClassifier`
   - `MultinomialNB`
6. Combine them with a **soft-voting ensemble** (`VotingClassifier`)  
7. Evaluate accuracy and classification report  
8. Save both:
   - Trained ensemble model  
   - TF-IDF vectorizer  

into a single file: `ensemble_model.pkl`

At app runtime, we only **load** this pickle and use it to predict sentiment for new tweets.
"""
    )
