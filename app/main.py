import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway
import plotly.express as px

# Constants
DATA_FILES = {
    'Benin': '../data/benin_clean.csv',
    'Sierra Leone': '../data/sierraleone_clean.csv',
    'Togo': '../data/togo_clean.csv'
}


NUMERIC_COLS = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust', 'Tamb', 'RH', 'WD', 'BP']

def load_data(country):
    df = pd.read_csv(DATA_FILES[country], parse_dates=['Timestamp'])
    return df

def summary_stats(df):
    st.subheader("Summary Statistics (Numeric Columns)")
    st.dataframe(df.describe())
    st.subheader("Missing Values")
    missing = df.isna().sum()
    st.dataframe(missing)
    cols_with_nulls = missing[missing > 0]
    st.write(f"Columns with missing values: {list(cols_with_nulls.index)}")
    cols_gt_5pct = missing[missing > 0.05 * len(df)]
    if not cols_gt_5pct.empty:
        st.warning(f"Columns with >5% missing values: {list(cols_gt_5pct.index)}")

def detect_outliers(df):
    st.subheader("Outlier Detection using Z-scores (|Z| > 3)")
    z_scores = df[NUMERIC_COLS].apply(zscore)
    outlier_flags = (np.abs(z_scores) > 3)
    st.write(f"Number of outliers per column:")
    st.dataframe(outlier_flags.sum())
    return outlier_flags

def clean_data(df, outlier_flags):
    # Impute median for missing in key columns
    key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
    for col in key_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    # Optionally drop rows with outliers or replace with median
    for col in key_cols:
        median_val = df[col].median()
        df.loc[outlier_flags[col], col] = median_val
    return df

def plot_time_series(df, country):
    st.subheader(f"Time Series Plots for {country}")
    cols_to_plot = ['GHI', 'DNI', 'DHI', 'Tamb']
    fig, axs = plt.subplots(len(cols_to_plot), 1, figsize=(10, 8), sharex=True)
    for i, col in enumerate(cols_to_plot):
        axs[i].plot(df['Timestamp'], df[col])
        axs[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

def cleaning_impact_plot(df):
    if 'Cleaning' in df.columns:
        st.subheader("Impact of Cleaning on ModA & ModB")
        grouped = df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
        fig, ax = plt.subplots()
        grouped.plot(kind='bar', x='Cleaning', y=['ModA', 'ModB'], ax=ax)
        st.pyplot(fig)
    else:
        st.info("No 'Cleaning' column to analyze cleaning impact.")

def correlation_heatmap(df):
    st.subheader("Correlation Heatmap")
    corr = df[['GHI', 'DNI', 'DHI', 'ModA', 'ModB']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def scatter_plots(df):
    st.subheader("Scatter Plots")
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].scatter(df['WS'], df['GHI'])
    axs[0, 0].set_xlabel('WS'); axs[0, 0].set_ylabel('GHI')

    axs[0, 1].scatter(df['WSgust'], df['GHI'])
    axs[0, 1].set_xlabel('WSgust'); axs[0, 1].set_ylabel('GHI')

    axs[1, 0].scatter(df['WD'], df['GHI'])
    axs[1, 0].set_xlabel('WD'); axs[1, 0].set_ylabel('GHI')

    axs[1, 1].scatter(df['RH'], df['Tamb'])
    axs[1, 1].set_xlabel('RH'); axs[1, 1].set_ylabel('Tamb')

    plt.tight_layout()
    st.pyplot(fig)

def wind_analysis(df):
    st.subheader("Wind Speed Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['WS'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

def bubble_chart(df):
    st.subheader("Bubble Chart: GHI vs Tamb (size=RH)")
    fig = px.scatter(df, x='GHI', y='Tamb', size='RH', hover_data=['Timestamp'], title='GHI vs Tamb with RH as bubble size')
    st.plotly_chart(fig)

def kpis(df):
    st.subheader("Key Performance Indicators (KPIs)")
    st.metric("Average GHI", f"{df['GHI'].mean():.2f}")
    st.metric("Average DNI", f"{df['DNI'].mean():.2f}")
    st.metric("Average DHI", f"{df['DHI'].mean():.2f}")

def cross_country_comparison():
    st.title("Cross-Country Comparison")
    dfs = {c: load_data(c) for c in DATA_FILES.keys()}
    df_all = pd.concat([df.assign(Country=country) for country, df in dfs.items()])
    
    metric = st.selectbox("Select metric to compare", ['GHI', 'DNI', 'DHI'])
    
    st.subheader(f"Boxplot of {metric} by Country")
    fig = px.box(df_all, x='Country', y=metric, points="all", color='Country')
    st.plotly_chart(fig)

    st.subheader("Summary Statistics by Country")
    summary = df_all.groupby('Country')[metric].agg(['mean','median','std']).reset_index()
    st.dataframe(summary)

    # Optional ANOVA
    if st.checkbox("Run ANOVA test"):
        groups = [group[metric].dropna() for name, group in df_all.groupby('Country')]
        stat, p = f_oneway(*groups)
        st.write(f"ANOVA F-statistic: {stat:.4f}, p-value: {p:.4e}")
        if p < 0.05:
            st.success("Statistically significant differences detected!")
        else:
            st.info("No statistically significant differences detected.")

    st.subheader("Average GHI Ranking")
    ranking = summary.sort_values('mean', ascending=False)
    fig2 = px.bar(ranking, x='Country', y='mean', title='Countries ranked by average GHI')
    st.plotly_chart(fig2)

def main():
    st.title("Solar Data Insights Dashboard")

    page = st.sidebar.selectbox("Choose a page", ["Country Analysis", "Cross-Country Comparison"])
    
    if page == "Country Analysis":
        country = st.sidebar.selectbox("Select Country", list(DATA_FILES.keys()))
        df = load_data(country)
        summary_stats(df)
        outlier_flags = detect_outliers(df)
        df_cleaned = clean_data(df, outlier_flags)
        plot_time_series(df_cleaned, country)
        cleaning_impact_plot(df_cleaned)
        correlation_heatmap(df_cleaned)
        scatter_plots(df_cleaned)
        wind_analysis(df_cleaned)
        bubble_chart(df_cleaned)
        kpis(df_cleaned)
    else:
        cross_country_comparison()

if __name__ == "__main__":
    main()
