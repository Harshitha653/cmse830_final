import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_stock_trends(df: pd.DataFrame, ticker: str):
    """Line chart of close price over time for a given ticker."""
    df_t = df[df["ticker"] == ticker]
    if df_t.empty:
        st.warning(f"No data found for ticker {ticker}.")
        return
    fig = px.line(
        df_t,
        x="date",
        y="close",
        title=f"Stock Close Price – {ticker}",
        labels={"date": "Date", "close": "Close price"},
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment_distribution(df: pd.DataFrame):
    """Bar chart of tweet sentiment counts."""
    if "airline_sentiment" not in df.columns:
        st.warning("Tweets dataset missing 'airline_sentiment' column.")
        return px.bar()
    counts = df["airline_sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(
        counts,
        x="sentiment",
        y="count",
        title="Tweet Sentiment Distribution",
        labels={"sentiment": "Sentiment", "count": "Count"},
    )
    return fig


def plot_news_sentiment(df: pd.DataFrame):
    """Line chart of number of news items per day."""
    if "publish_date" not in df.columns:
        st.warning("News dataset missing 'publish_date' column.")
        return
    df_g = df.groupby("publish_date").size().reset_index(name="count")
    fig = px.line(
        df_g,
        x="publish_date",
        y="count",
        title="News Count Over Time",
        labels={"publish_date": "Date", "count": "Number of headlines"},
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud(df: pd.DataFrame, column: str):
    """Generate and display a word cloud from a text column."""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found for word cloud.")
        return
    text = " ".join(df[column].dropna().astype(str).tolist())
    if not text.strip():
        st.warning("No text available to build a word cloud.")
        return

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
