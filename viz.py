import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

sns.set_theme(style="whitegrid", font_scale=1.1)

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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric].corr(), cmap="viridis")
    st.pyplot(fig)


def plot_pairplot(df, cols):
    g = sns.pairplot(df[cols].dropna())
    st.pyplot(g.fig)


def plot_time_series(df, time_col, val_col):
    fig, ax = plt.subplots()
    df = df.dropna(subset=[time_col, val_col]).sort_values(time_col)
    ax.plot(df[time_col], df[val_col])
    plt.xticks(rotation=45)
    st.pyplot(fig)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_heatmap(df):
    missing = df.isnull()
    plt.figure(figsize=(10, 5))
    sns.heatmap(missing, cbar=False, cmap="Reds")
    st.pyplot(plt)

def plot_four_panel(df, col_a, col_x, col_y):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))  # wider
    axes = axes.flatten()

    # Histogram
    sns.histplot(df[col_a].dropna(), ax=axes[0], kde=True)
    axes[0].set_title(f"Histogram of {col_a}")
    axes[0].tick_params(axis='x', rotation=0)

    # Boxplot
    sns.boxplot(x=df[col_a], ax=axes[1])
    axes[1].set_title(f"Boxplot of {col_a}")
    axes[1].tick_params(axis='x', rotation=0)

    # Scatter
    sns.scatterplot(x=df[col_x], y=df[col_y], ax=axes[2], s=15)
    axes[2].set_title(f"{col_x} vs {col_y}")
    axes[2].tick_params(axis='x', rotation=0)

    # Local correlation heatmap
    corr = df[[col_a, col_x, col_y]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[3], cbar=True)
    axes[3].set_title("Local Correlations")

    plt.tight_layout()
    st.pyplot(fig)
