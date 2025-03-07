import pandas as pd
import numpy as np
import mlflow
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse

def load_data(file_path):
    """loading  customer purchases data."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Processed data loaded successfully!")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def perform_clustering(df):
    """Computing RFM metrics, normalize data, apply K-Means clustering, evaluate, and visualize."""
    try:
        df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])
        current_date = df['LastPurchaseDate'].max()

        rfm = df.groupby('CustomerID').agg(
            Recency=('LastPurchaseDate', lambda x: (current_date - x.max()).days),
            Frequency=('StockCode', 'count'),
            Monetary=('TotalQuantity', 'sum')
        ).reset_index()
        
        rfm_normalized = (rfm[['Recency', 'Frequency', 'Monetary']] - rfm[['Recency', 'Frequency', 'Monetary']].min()) / \
                         (rfm[['Recency', 'Frequency', 'Monetary']].max() - rfm[['Recency', 'Frequency', 'Monetary']].min())

        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_normalized)

        silhouette_avg = silhouette_score(rfm_normalized, rfm['Cluster'])
        davies_bouldin = davies_bouldin_score(rfm_normalized, rfm['Cluster'])
        
        logging.info(f"Silhouette Score: {silhouette_avg}")
        logging.info(f"Davies-Bouldin Index: {davies_bouldin}")
        
        # Save clustered data
        rfm.to_csv('customer_segments.csv', index=False)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='viridis')
        plt.title('Customer Segmentation (Recency vs Monetary)')
        plt.xlabel('Recency (days)')
        plt.ylabel('Monetary (total quantity)')
        plt.savefig("customer_segmentation.png")
        
        return silhouette_avg, davies_bouldin, 'customer_segments.csv'
    except Exception as e:
        logging.error(f"Error in clustering: {e}")
        return None, None, None

def train_recommendation_model(df):
    """Train a collaborative filtering model using SVD."""
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['CustomerID', 'StockCode', 'TotalQuantity']], reader)

        svd = SVD()
        cross_validate(svd, data, cv=5, verbose=True)
        
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        
        testset = trainset.build_testset()
        predictions = svd.test(testset)
        rmse_value = rmse(predictions)
        
        logging.info(f"RMSE for Recommendation Model: {rmse_value}")
        return rmse_value
    except Exception as e:
        logging.error(f"Error in recommendation model training: {e}")
        return None

def main():
    mlflow.set_experiment("customer_segmentation_and_recommendation")
    with mlflow.start_run():
        file_path = 'processed_customer_purchases.csv'
        df = load_data(file_path)
        
        if df is not None:
            silhouette_avg, davies_bouldin, clustered_data_path = perform_clustering(df)
            rmse_value = train_recommendation_model(df)
            
            # Logging artifacts and metrics in MLflow
            mlflow.log_artifact(clustered_data_path)
            mlflow.log_artifact("customer_segmentation.png")
            mlflow.log_metric("silhouette_score", silhouette_avg)
            mlflow.log_metric("davies_bouldin_index", davies_bouldin)
            mlflow.log_metric("rmse", rmse_value)
            
            logging.info("MLflow tracking complete!")

if __name__ == "__main__":
    main()
