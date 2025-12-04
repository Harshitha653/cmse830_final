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
# 1. LOAD LOCAL DATA FILES FROM GITHUB
# -------------------------------------------------------
@st.cache_data
def load_data():
    # Load your uploaded local files
    news1 = pd.read_csv("data/news_tiny_part1.csv")
    news2 = pd.read_csv("data/news_tiny_part2.csv")
    df_news = pd.concat([news1, news2], ignore_index=True)

    # Load tweets (you must upload Tweets.csv to /data/)
    df_tweets = pd.read_csv("data/Tweets.csv")

    # OPTIONAL: Remove stocks if unused
    # But if you want synthetic or placeholder stocks, uncomment:
    # df_stocks = generate_synthetic_stocks(df_tweets)
    # return df_stocks, df_tweets, df_news

    # For now: remove stocks page by using synthetic empty frame
    df_stocks = pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=30),
        "open": 100,
        "high": 101,
        "low": 99,
        "close": 100,
        "volume": 100000,
        "ticker": "SYNTH",
        "daily_return": 0,
        "volatility": 0
    })

    # Preprocess
    df_tweets = preprocess_tweets(df_tweets)
    df_news = preprocess_news(df_news)

    # Map tweet sentiment
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
    ["Overview", "Tweets", "News", "Sentiment Prediction", "Modeling Details"],
)


# -------------------------------------------------------
# 3. OVERVIEW PAGE
# -------------------------------------------------------
if page == "Overview":
    st.title("📊 Market & Sentiment Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tweets sample")
        st.dataframe(df_tweets.head())

    with col2:
        st.subheader("News sample")
        st.dataframe(df_news.head())


# -------------------------------------------------------
# 4. TWEETS PAGE
# -------------------------------------------------------
elif page == "Tweets":
    st.title("💬 Tweets Sentiment Analysis")

    st.subheader("Sentiment distribution")
    fig = plot_sentiment_distribution(df_tweets)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tweet word cloud")
    plot_wordcloud(df_tweets, "text_clean")

    st.subheader("Sample tweets")
    st.dataframe(df_tweets[["text", "airline_sentiment", "sentiment_label"]].head(20))


# -------------------------------------------------------
# 5. NEWS PAGE
# -------------------------------------------------------
elif page == "News":
    st.title("📰 News Headline Analysis")

    st.subheader("News count over time")
    plot_news_sentiment(df_news)

    st.subheader("News word cloud")
    plot_wordcloud(df_news, "headline_text_clean")

    st.subheader("Sample headlines")
    st.dataframe(df_news.head(20))


# -------------------------------------------------------
# 6. SENTIMENT PREDICTION PAGE
# -------------------------------------------------------
elif page == "Sentiment Prediction":
    st.title("🔮 Tweet Sentiment Prediction")

    text = st.text_area("Enter a tweet:")

    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model("ensemble_model_compact.pkl.gz")

    bundle = st.session_state.model_bundle
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    if st.button("Predict sentiment"):
        if text.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            pred = predict_sentiment(bundle, vectorizer, text)
            label_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
            st.success(f"Sentiment: **{label_map.get(pred, 'Unknown')}**")


# -------------------------------------------------------
# 7. MODELING DETAILS PAGE
# -------------------------------------------------------
elif page == "Modeling Details":
    st.title("🧠 Modeling & Data Pipeline Details")

    st.markdown(
        """
### 📌 Datasets Used

**Tweets Dataset (Airline Sentiment):**  
- Contains text, label, and timestamp.  
- Used for **training & prediction**.

**News Dataset (Split Into Two Files):**  
- Large ABC News headlines dataset split into:
  - `news_tiny_part1.csv`
  - `news_tiny_part2.csv`
- Combined and preprocessed inside the app.

---

### 🧼 Preprocessing Steps

#### Tweets:
- Lowercasing  
- Remove URLs, mentions  
- Remove punctuation  
- Stopword removal (scikit-learn)  
- Sentiment mapping:  
  - positive → 1  
  - neutral → 0  
  - negative → -1  

#### News:
- Clean headline text  
- Convert YYYYMMDD → datetime  
- Assign cleaned version to `headline_text_clean`  

---

### 🤖 Sentiment Model (ensemble_model.pkl)

- TF-IDF vectorizer (1–2 grams, max 5000 features)
- RandomForestClassifier
- Multinomial Naive Bayes
- VotingClassifier (soft)
"""
    )
