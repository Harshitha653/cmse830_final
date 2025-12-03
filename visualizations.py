# visualizations.py
import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_stock_trends(df, ticker='a.us'):
    df_ticker = df[df['Ticker'] == ticker]
    fig = px.line(df_ticker, x='Date', y='Close', title=f'Stock Price Trend for {ticker}')
    st.plotly_chart(fig)

import plotly.express as px

def plot_sentiment_distribution(df):
    # count sentiments
    counts = df['airline_sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment', 'count']   # rename properly

    fig = px.bar(
        counts,
        x='sentiment',
        y='count',
        labels={'sentiment': 'Sentiment', 'count': 'Count'},
        title='Tweet Sentiment Distribution'
    )
    return fig


def plot_news_sentiment(df):
    df_grouped = df.groupby('publish_date').size().reset_index(name='count')
    fig = px.line(df_grouped, x='publish_date', y='count', title='News Count Over Time')
    st.plotly_chart(fig)

def plot_scatter_sentiment_vs_stock(df_stock, df_tweets):
    merged = pd.merge(df_stock, df_tweets, left_on='Date', right_on='tweet_created', how='inner')
    fig = px.scatter(merged, x='Daily_Return', y='sentiment_label', color='Ticker', title='Stock Daily Return vs. Tweet Sentiment')
    st.plotly_chart(fig)

def plot_wordcloud(df, column='text_clean'):
    text = " ".join(df[column].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
