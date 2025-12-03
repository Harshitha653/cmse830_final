# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import re
import string

# ✅ Replace NLTK stopwords with sklearn's built-in list (Cloud-safe)
stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_tweets(df):
    df.columns = df.columns.str.strip().str.lower()

    print("Tweet columns:", df.columns.tolist())

    text_col = "text"

    df["text_clean"] = df[text_col].astype(str).apply(clean_text)

    if "tweet_created" in df.columns:
        df["date"] = pd.to_datetime(df["tweet_created"], errors="coerce")

    return df

def preprocess_news(df):
    df['headline_text_clean'] = df['headline_text'].apply(clean_text)
    df['publish_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')
    return df

def preprocess_stocks(df):
    # Normalize column names → all lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "openint"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # ✔ Create Daily Return
    df["daily_return"] = df["close"].pct_change().fillna(0)

    # ✔ 7-day rolling volatility
    df["volatility"] = df["daily_return"].rolling(window=7).std().fillna(0)

    return df


def vectorize_text(df, column='text_clean', method='tfidf', max_features=5000):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[column])
    return X, vectorizer
