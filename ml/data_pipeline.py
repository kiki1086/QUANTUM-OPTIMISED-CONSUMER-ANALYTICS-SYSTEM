import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class DataPipeline:
    def __init__(self, data_dir="../datasets"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        self.customers = pd.read_csv(f"{self.data_dir}/customers.csv")
        self.products = pd.read_csv(f"{self.data_dir}/products.csv")
        self.transactions = pd.read_csv(f"{self.data_dir}/transactions.csv")
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        
    def clean_data(self):
        # Handle missing values (stub)
        self.transactions.dropna(inplace=True)
        # Outlier detection (stub)
        self.transactions = self.transactions[self.transactions['quantity'] > 0]
        
    def generate_features(self):
        """Generates features for both purchase prediction and demand forecasting"""
        # Transactions already contain some product information, avoid duplicates
        df = self.transactions.merge(self.customers, on='customer_id')
        
        # If 'category' is not in transactions but is in products, merge it (though our mock script already adds it)
        if 'category' not in df.columns and 'category' in self.products.columns:
             df = df.merge(self.products[['product_id', 'category']], on='product_id', how='left')
        
        # 1. RFM Features (Recency, Frequency, Monetary)
        max_date = df['date'].max()
        rfm = df.groupby('customer_id').agg({
            'date': lambda x: (max_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).rename(columns={'date': 'recency', 'transaction_id': 'frequency', 'total_amount': 'monetary'})
        
        # 2. Categorical Encoding
        for col in ['region', 'category']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Feature Store Population (stub)
        os.makedirs(f"{self.data_dir}/features", exist_ok=True)
        rfm.to_csv(f"{self.data_dir}/features/rfm_features.csv")
        
        self.master_df = df
        return df

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.load_data()
    pipeline.clean_data()
    df = pipeline.generate_features()
    print("Data Engineering Pipeline completed. Features generated.")
