from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()

    # rename columns
    df = df.rename(columns={'Genre':'Sex', 'Annual Income (k$)':'Annual_Income', 'Spending Score (1-100)':'Spending_Score'})
    
    # keep required columns
    req_cols = ['Annual_Income', 'Spending_Score']
    df=df[req_cols]
    
    df['Annual_Income'] = df['Annual_Income'] * 1000

    # Convert the Columns to Float Before Scaling
    cols = ['Annual_Income', 'Spending_Score']
    df.loc[:,cols] = df[cols].astype(float)

    return df

def feature_importance(X, n_clusters):
    X = X.copy()
    full_score = silhouette_score(X, KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X))
    scores = {}
    for col in X.columns:
        subset = X.drop(columns=[col])
        score = silhouette_score(subset, KMeans(n_clusters=n_clusters, random_state=42).fit_predict(subset))
        scores[col] = full_score - score
    return pd.Series(scores).sort_values(ascending=False)

