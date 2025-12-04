# data_processing.py
import numpy as np
import pandas as pd


# -------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------
def clean_dataframe(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    """
    Advanced cleaning:
    - strip column names
    - drop duplicates
    - trim string columns
    - try to parse any *date* column
    - fill missing numeric values with median
    """
    df = df.copy()

    df.columns = [c.strip() for c in df.columns]
    df.drop_duplicates(inplace=True)

    # string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # parse date-like columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass

    # numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


# -------------------------------------------------
# HELPER FUNCTIONS FOR AUTO-JOIN
# -------------------------------------------------
def _canonical(col: str) -> str:
    """Lowercase and remove non-alphanumeric → used to match column names."""
    return "".join(ch.lower() if ch.isalnum() else "" for ch in col)


def _build_canon_map(df: pd.DataFrame) -> dict:
    """Map canonical name → original name."""
    mapping = {}
    for col in df.columns:
        c = _canonical(col)
        if c and c not in mapping:
            mapping[c] = col
    return mapping


def _auto_merge_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    preferred_keys: list[str],
) -> pd.DataFrame:
    """
    Try to merge left & right on sensible common columns.
    - Prefer columns in preferred_keys (by canonical name)
    - If none, use any common canonical column
    - Always use OUTER join (to avoid 0 rows)
    - Cast join columns to string to avoid dtype issues
    - If absolutely no common columns, concat side-by-side
    """
    left_map = _build_canon_map(left)
    right_map = _build_canon_map(right)

    # 1) preferred keys (e.g., city, country, date)
    for key in preferred_keys:
        if key in left_map and key in right_map:
            lk = left_map[key]
            rk = right_map[key]
            L = left.copy()
            R = right.copy()
            L[lk] = L[lk].astype(str)
            R[rk] = R[rk].astype(str)
            merged = pd.merge(L, R, left_on=lk, right_on=rk, how="outer")
            return merged

    # 2) any common column
    common = set(left_map.keys()) & set(right_map.keys())
    if common:
        canon = sorted(common)[0]
        lk = left_map[canon]
        rk = right_map[canon]
        L = left.copy()
        R = right.copy()
        L[lk] = L[lk].astype(str)
        R[rk] = R[rk].astype(str)
        merged = pd.merge(L, R, left_on=lk, right_on=rk, how="outer")
        return merged

    # 3) fallback: no common columns → concat
    return pd.concat(
        [left.reset_index(drop=True), right.reset_index(drop=True)],
        axis=1,
    )


# -------------------------------------------------
# MAIN AUTO-INTEGRATION
# -------------------------------------------------
def integrate_data(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Automatically integrates your three datasets:

    dfs = [air_df, weather_df, pop_df]

    Steps:
    1. Merge Air + Weather by city / date / datetime (best available)
    2. Merge result + Population by country field (best available)
    3. Clean the final integrated dataset again

    No user interaction required.
    """
    if len(dfs) != 3:
        raise ValueError("integrate_data expects exactly 3 dataframes")

    air_df, weather_df, pop_df = dfs

    # 1) Air + Weather
    merged_aw = _auto_merge_two(
        air_df,
        weather_df,
        preferred_keys=[
            "city",
            "location",
            "station",
            "datetimeutc",
            "date",
        ],
    )

    # 2) + Population
    merged_all = _auto_merge_two(
        merged_aw,
        pop_df,
        preferred_keys=[
            "country",
            "countryterritory",
            "countryregion",
        ],
    )

    # 3) Final cleanup (numeric imputation etc.)
    merged_all = clean_dataframe(merged_all, name="Integrated")

    return merged_all


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    - log1p on numeric columns
    - z-score normalization
    - simple interaction terms among first few numeric columns
    """
    df = df.copy()
    num_cols = get_numeric_columns(df)

    # log features
    for col in num_cols:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # z-score features
    for col in num_cols:
        std = df[col].std()
        if std == 0 or np.isnan(std):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / std

    # interactions: first 5 numeric columns only
    base = num_cols[:5]
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            c1, c2 = base[i], base[j]
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    return df
