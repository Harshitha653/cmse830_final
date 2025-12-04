import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------------------------------------
# STOCK TRENDS
# ---------------------------------------------
def plot_stock_trends(df, ticker):
    df_t = df[df['ticker'] == ticker]
    fig = px.line(df_t, x='date', y='close',
                  title=f'Stock Price Trend: {ticker.upper()}')
    st.plotly_chart(fig)

# ---------------------------------------------
# TWEET SENTIMENT DISTRIBUTION
# ---------------------------------------------
def plot_sentiment_distribution(df):
    counts = df['airline_sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment', 'count']
    fig = px.bar(counts, x='sentiment', y='count',
                 title='Tweet Sentiment Distribution')
    return fig

# ---------------------------------------------
# NEWS SENTIMENT OVER TIME
# ---------------------------------------------
def plot_news_sentiment(df):
    df_g = df.groupby('publish_date').size().reset_index(name='count')
    fig = px.line(df_g, x='publish_date', y='count',
                  title='News Count Over Time')
    st.plotly_chart(fig)

# ---------------------------------------------
# WORD CLOUD
# ---------------------------------------------
def plot_wordcloud(df, column):
    text = " ".join(df[column].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(15,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
