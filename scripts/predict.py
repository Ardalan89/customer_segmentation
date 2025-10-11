import argparse
import os
import pandas as pd
from joblib import load
from scripts.helpers import clean_data
from sklearn.metrics import pairwise_distances_argmin_min

def main(model_type):
    # choose model based on argument
    if model_type.lower() == 'kmeans':
        model_path = './models/model_Kmeans_cluster.joblib'
    elif model_type.lower() == 'hc':
        model_path = './models/model_hc_cluster.joblib'
    else:
        raise ValueError("Invalid model type. Use 'kmeans' or 'hc'.")

    print(f"Loading model from: {model_path}")
    if model_type.lower() == 'kmeans': 
        pipeline = load(model_path)
    else:
        pipeline, cluster_centroids = load(model_path)

    # load new data 
    new_data = pd.read_csv('./data/raw/New_Mall_Customer.csv')

    # clean data
    X = clean_data(new_data)

    # predict cluster labels
    if model_type.lower() == 'kmeans':
        y_pred = pipeline.predict(X)
    else:  
        X_scaled = pipeline.named_steps['preprocess'].transform(X)
        # assign new samples to nearest cluster centroid
        closest, _ = pairwise_distances_argmin_min(X_scaled, cluster_centroids)
        y_pred = closest

    # create output DataFrame
    pred_df = new_data.copy()
    pred_df['customer_cluster_' + model_type.lower()] = y_pred

    # output file path
    pred_path = './data/predict_new_customer_cluster.csv'

    # save or update the file
    if not os.path.exists(pred_path):
        pred_df.to_csv(pred_path, index=False)
        print(f" Created new file: {pred_path}")
    else:
        existing = pd.read_csv(pred_path)
        if len(existing) == len(y_pred):
            existing['customer_cluster_' + model_type.lower()] = y_pred
            existing.to_csv(pred_path, index=False)
            print(f"Updated existing file: {pred_path}")
        else:
            print("Warning: Row counts differ — not updating the file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict clusters using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Type of model: 'kmeans' or 'hc'")
    args = parser.parse_args()

    main(args.model)