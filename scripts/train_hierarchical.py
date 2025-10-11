from joblib import dump
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

from scripts.helpers import clean_data
from scripts.preprocess import full_processor

# load data
df = pd.read_csv("./data/raw/Mall_Customers.csv")

# Clean the data
X = clean_data(df)


# build full model pipeline 
pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', AgglomerativeClustering(n_clusters = 5, metric= 'euclidean', linkage = 'ward'))
])

# preprocess data 
X_scaled = pipeline.named_steps['preprocess'].fit_transform(X)

# train + predict clusters
model = pipeline.named_steps['model']
y_pred = model.fit_predict(X_scaled)

# compute cluster centroids (mean per cluster)
import numpy as np
cluster_centroids = np.array([
    X_scaled[y_pred == i].mean(axis=0)
    for i in np.unique(y_pred)
])

# evaluate model
score = silhouette_score(X_scaled, y_pred)
print(f"Silhouette Score: {score:.3f}")

# Save trained model + cluster centroid
dump((pipeline, cluster_centroids), './models/model_hc_cluster.joblib')
print("Saved model to './models/model_hc_cluster.joblib'")