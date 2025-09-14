import sys
import os

# Add the directory containing app.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from analysis import detect_outliers_iqr, detect_outliers_isolation_forest, detect_outliers_dbscan
from visualization import plot_box, plot_interactive

st.set_page_config(page_title="Outlier Detection App", layout="wide")
st.title("📊 Outlier Detection and Analysis")

uploaded_file = st.file_uploader("Upload your dataset (.csv, .xls, .xlsx)", type=['csv', 'xls', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found!")
    else:
        selected_col = st.selectbox("Select column to analyze", numeric_cols)
        st.subheader(f"Analyzing {selected_col}")

        algorithm = st.radio("Choose Outlier Detection Method", ["IQR", "Isolation Forest", "DBSCAN"])

        if algorithm == "IQR":
            outliers = detect_outliers_iqr(df[selected_col])
        elif algorithm == "Isolation Forest":
            contamination = st.slider("Contamination fraction", 0.01, 0.2, 0.05)
            outliers = detect_outliers_isolation_forest(df[selected_col], contamination)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 1.5, 0.5)
            min_samples = st.slider("Minimum samples", 1, 10, 5)
            outliers = detect_outliers_dbscan(df[selected_col], eps, min_samples)

        st.write(f"Total data points: {len(df)}")
        st.write(f"Outliers detected: {len(outliers)}")

        st.subheader("Box Plot")
        plot_box(df, selected_col)

        st.subheader("Interactive Plot")
        plot_interactive(df, selected_col)

        if not outliers.empty:
            st.subheader("Outliers")
            st.dataframe(outliers)
            csv = outliers.to_csv(index=False).encode()
            st.download_button("Download Outliers as CSV", data=csv, file_name="outliers.csv")
