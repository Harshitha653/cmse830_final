import numpy as np
import pandas as pd

def clean_dataframe(df, name="Dataset"):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.drop_duplicates(inplace=True)

    # Clean string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Parse date columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="ignore")

    # Fill missing values
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def integrate_data(dfs, join_keys, how="inner"):
    df0, df1, df2 = dfs
    jk0, jk1, jk2 = join_keys

    merged = df0.merge(
        df1, left_on=jk0, right_on=jk1, how=how, suffixes=("", "_w")
    )
    merged = merged.merge(
        df2, left_on=jk0, right_on=jk2, how=how, suffixes=("", "_p")
    )
    return merged


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def add_feature_engineering(df):
    df = df.copy()
    numeric = get_numeric_columns(df)

    # Log transform
    for col in numeric:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Z-score normalization
    for col in numeric:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    # Feature interactions (first 5 numeric)
    base = numeric[:5]
    for i in range(len(base)):
        for j in range(i+1, len(base)):
            df[f"{base[i]}_x_{base[j]}"] = df[base[i]] * df[base[j]]

    return df
