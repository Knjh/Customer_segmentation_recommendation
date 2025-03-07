import pandas as pd
import mlflow
import logging

def load_data(file_path):
    """loading dataset from the given file path."""
    try:
        df = pd.read_excel(file_path)
        logging.info("Data loaded successfully!")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_and_save_data(df, output_path):
    """Cleaning dataset, process customer purchases, and save the output."""
    try:
        df.dropna(subset=['CustomerID'], inplace=True)
        df.fillna({'Description': 'Unknown'}, inplace=True)

        df.drop_duplicates(inplace=True)

        # data type coversion
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['CustomerID'] = df['CustomerID'].astype(int)

        # customer Purchases
        customer_purchases = df.groupby(['CustomerID', 'StockCode']).agg(
            {'Quantity': 'sum', 'InvoiceDate': 'max'}
        ).reset_index()
        customer_purchases.rename(columns={'Quantity': 'TotalQuantity', 'InvoiceDate': 'LastPurchaseDate'}, inplace=True)
        
        customer_purchases.to_csv(output_path, index=False)
        logging.info("Data preprocessing complete! Processed data saved.")
    
    except Exception as e:
        logging.error(f"Error processing data: {e}")

def main():
    # mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment("customer_data_processing")
    with mlflow.start_run():
        file_path = 'Online Retail.xlsx'
        output_path = 'processed_customer_purchases.csv'
        
        df = load_data(file_path)
        if df is not None:
            clean_and_save_data(df, output_path)
            mlflow.log_artifact(output_path)  
            logging.info("MLflow tracking complete!")

if __name__ == "__main__":
    main()
