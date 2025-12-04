import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def show_basic_info(df, name):
    st.write(f"### {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())
    st.write("Column Types")
    st.dataframe(df.dtypes.to_frame())
    st.write("Summary Statistics")
    st.dataframe(df.describe().T)


def plot_numeric_distribution(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)


def plot_boxplot(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)


def plot_scatter(df, x, y):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    st.pyplot(fig)


def plot_correlation_heatmap(df, numeric):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[numeric].corr(), cmap="viridis")
    st.pyplot(fig)


def plot_pairplot(df, cols):
    g = sns.pairplot(df[cols].dropna())
    st.pyplot(g.fig)


def plot_time_series(df, time_col, val_col):
    fig, ax = plt.subplots()
    df = df.dropna(subset=[time_col, val_col])
    df = df.sort_values(time_col)
    ax.plot(df[time_col], df[val_col])
    plt.xticks(rotation=45)
    st.pyplot(fig)
