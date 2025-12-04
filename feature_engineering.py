import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import re
import string

# Cloud-safe stopwords (no NLTK download)
stop_words = ENGLISH_STOP_WORDS


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|\#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "text" not in df.columns:
        raise ValueError("Tweets dataset must contain a 'text' column.")

    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].apply(clean_text)

    if "tweet_created" in df.columns:
        df["date"] = pd.to_datetime(df["tweet_created"], errors="coerce")

    return df


def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "headline_text" not in df.columns or "publish_date" not in df.columns:
        raise ValueError("News dataset must have 'headline_text' and 'publish_date' columns.")

    df["headline_text"] = df["headline_text"].astype(str)
    df["headline_text_clean"] = df["headline_text"].apply(clean_text)

    # Try flexible parse for YYYYMMDD or general date formats
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce", format="%Y%m%d")

    return df


def preprocess_stocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("Stocks dataset must contain a 'Date' column.")

    # Date parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Numeric columns (only those present)
    numeric_candidates = ["open", "high", "low", "close", "volume", "openint"]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Feature engineering
    if "close" in df.columns:
        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["volatility"] = df["daily_return"].rolling(window=7).std().fillna(0)
    else:
        df["daily_return"] = 0.0
        df["volatility"] = 0.0

    return df


def vectorize_text(df: pd.DataFrame, column: str = "text_clean", method: str = "tfidf", max_features: int = 5000):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[column].astype(str))
    return X, vectorizer
