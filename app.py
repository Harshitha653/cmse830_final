# app.py
import os
import streamlit as st
import pandas as pd

from data_processing import (
    clean_dataframe,
    integrate_data,
    add_feature_engineering,
    get_numeric_columns,
)
from modeling import prepare_model_data, train_models, evaluate_models
from viz import (
    show_basic_info,
    plot_numeric_distribution,
    plot_boxplot,
    plot_scatter,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_time_series,
)

st.set_page_config(
    page_title="CMSE 830 – Multi-Source Data Science Dashboard",
    layout="wide",
)

DATA_DIR = "data"


@st.cache_data
def load_local_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_all_datasets():
    """
    Uses the exact files in your /data folder.
    Update names here if your CSV names change.
    """
    air = load_local_csv("global_air_quality_data_10000.csv")
    weather = load_local_csv("testset.csv")          # Delhi / weather CSV
    pop = load_local_csv("world_population.csv")
    return air, weather, pop


def main():
    st.title("🌍 CMSE 830 – Multi-Source Data Science Dashboard")

    st.markdown(
        """
        This app:
        - Loads **3 Kaggle datasets** from `/data`
        - Cleans & **integrates** them automatically
        - Performs advanced **EDA and visualization**
        - Builds & compares **ML models**
        - Uses advanced **Streamlit features** (caching, tabs, etc.)
        """
    )

    # ------------------ Load & clean ------------------
    with st.spinner("Loading and cleaning datasets..."):
        air_df, weather_df, pop_df = load_all_datasets()
        air_df = clean_dataframe(air_df, "Air Quality")
        weather_df = clean_dataframe(weather_df, "Weather")
        pop_df = clean_dataframe(pop_df, "Population")

        integrated_df = integrate_data([air_df, weather_df, pop_df])
        st.session_state["integrated_df"] = integrated_df

    st.success(f"Integrated dataset shape: {integrated_df.shape[0]} rows × {integrated_df.shape[1]} columns")

    # ------------------ Tabs ------------------
    tab_overview, tab_data, tab_eda, tab_model, tab_docs = st.tabs(
        ["Overview", "Data & Integration", "EDA", "Feature Engineering & Modeling", "Documentation"]
    )

    # ------------------ Overview ------------------
    with tab_overview:
        st.header("📘 Project Overview")
        st.markdown(
            """
            - **Data Sources**: Global Air Quality, Weather (Delhi), World Population  
            - **Integration**: Automatic joins on common fields (city / date / country)  
            - **Goal**: Explore relationships between pollution, climate, and population,
              and build predictive models.  
            """
        )

    # ------------------ Data & Integration ------------------
    with tab_data:
        st.header("🧩 Data & Integration")

        st.subheader("Air Quality Dataset")
        show_basic_info(air_df, "Air Quality")

        st.subheader("Weather Dataset")
        show_basic_info(weather_df, "Weather")

        st.subheader("Population Dataset")
        show_basic_info(pop_df, "Population")

        st.subheader("Integrated Dataset (Auto-joined)")
        show_basic_info(integrated_df, "Integrated")

    # ------------------ EDA ------------------
    with tab_eda:
        st.header("📊 Exploratory Data Analysis (Integrated Dataset)")

        df = integrated_df
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.error("No numeric columns found for EDA.")
        else:
            col_a = st.selectbox("Column for histogram / boxplot", numeric_cols)
            col_x = st.selectbox("X-axis (scatter)", numeric_cols)
            col_y = st.selectbox("Y-axis (scatter)", numeric_cols, index=min(1, len(numeric_cols) - 1))

            st.subheader("Histogram")
            plot_numeric_distribution(df, col_a)

            st.subheader("Boxplot")
            plot_boxplot(df, col_a)

            st.subheader("Scatter Plot")
            plot_scatter(df, col_x, col_y)

            st.subheader("Correlation Heatmap")
            plot_correlation_heatmap(df, numeric_cols)

            st.subheader("Pairplot")
            pair_cols = st.multiselect(
                "Select up to 5 columns for pairplot",
                numeric_cols,
                default=numeric_cols[: min(5, len(numeric_cols))],
            )
            if pair_cols:
                plot_pairplot(df, pair_cols)

            st.subheader("Time-Series Plot (if date columns exist)")
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                ts_col = st.selectbox("Date/Time column", date_cols)
                val_col = st.selectbox("Value column", numeric_cols)
                plot_time_series(df, ts_col, val_col)
            else:
                st.info("No obvious date/time columns detected in the integrated dataset.")

    # ------------------ Feature Engineering & Modeling ------------------
    with tab_model:
        st.header("🤖 Feature Engineering & Modeling")

        df = integrated_df
        engineered_df = add_feature_engineering(df)
        st.write(f"Engineered dataset shape: {engineered_df.shape}")
        st.dataframe(engineered_df.head())

        numeric_cols = get_numeric_columns(engineered_df)
        if not numeric_cols:
            st.error("No numeric columns available for modeling.")
        else:
            target_col = st.selectbox("Target variable (y)", numeric_cols)
            feature_candidates = [c for c in numeric_cols if c != target_col]
            default_features = feature_candidates[: min(10, len(feature_candidates))]

            feature_cols = st.multiselect(
                "Feature columns (X)",
                feature_candidates,
                default=default_features,
            )

            test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("CV folds", 3, 10, 5, 1)

            if st.button("Train & Evaluate Models"):
                if not feature_cols:
                    st.warning("Select at least one feature.")
                else:
                    X_train, X_test, y_train, y_test = prepare_model_data(
                        engineered_df, target_col, feature_cols, test_size=test_size
                    )
                    models = train_models(X_train, y_train)
                    results = evaluate_models(models, X_train, y_train, X_test, y_test, cv_folds=cv_folds)
                    st.subheader("Model Comparison")
                    st.dataframe(results)

    # ------------------ Documentation ------------------
    with tab_docs:
        st.header("📄 Documentation")
        st.markdown(
            """
            ### Datasets
            - **Global Air Quality Dataset** (Kaggle)  
            - **Delhi Climate / Weather Dataset** (Kaggle)  
            - **World Population Dataset** (Kaggle)  

            ### Rubric Mapping
            - 3 distinct data sources → ✔  
            - Advanced cleaning & integration → ✔  
            - Multiple visualizations (hist, box, scatter, heatmap, pairplot, time-series) → ✔  
            - Feature engineering (log, z-score, interactions) → ✔  
            - 2 ML models (Random Forest, Gradient Boosting) + CV → ✔  
            - Streamlit app with caching & tabs → ✔  
            """
        )


if __name__ == "__main__":
    main()
