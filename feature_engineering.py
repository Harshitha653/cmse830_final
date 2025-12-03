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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=7).std()

    df["Daily_Return"].fillna(0, inplace=True)
    df["Volatility"].fillna(0, inplace=True)

    return df

def vectorize_text(df, column='text_clean', method='tfidf', max_features=5000):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[column])
    return X, vectorizer
