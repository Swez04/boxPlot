import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data < lower) | (data > upper)]
    return outliers

def detect_outliers_isolation_forest(data, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data.values.reshape(-1, 1))
    preds = model.predict(data.values.reshape(-1, 1))
    outliers = data[preds == -1]
    return outliers

def detect_outliers_dbscan(data, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(data_scaled)
    outliers = data[model.labels_ == -1]
    return outliers