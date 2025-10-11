from joblib import dump
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from scripts.helpers import clean_data
from scripts.preprocess import full_processor

# load data
df = pd.read_csv("./data/raw/Mall_Customers.csv")

# Clean the data
X = clean_data(df)


# build full model pipeline 
pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', KMeans(n_clusters = 5, init = 'k-means++', random_state = 42))
])

# train
pipeline.fit(X)

# evaluate model
y_pred = pipeline.predict(X)

kmeans = pipeline.named_steps['model']
X_processed = pipeline.named_steps['preprocess'].transform(X)
score = silhouette_score(X_processed, kmeans.labels_)

print(f"Silhouette Score: {score:.3f}")


# Save trained pipeline
dump(pipeline, './models/model_Kmeans_cluster.joblib')
print("Saved model to './models/model_Kmeans_cluster.joblib'")