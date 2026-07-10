import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_ecommerce_data(num_customers=5000, num_products=200, days=365):
    np.random.seed(42)
    
    # 1. Generate Customers
    customer_ids = np.arange(1, num_customers + 1)
    ages = np.random.normal(loc=35, scale=12, size=num_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=num_customers)
    
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'region': regions
    })
    
    # 2. Generate Products
    product_ids = np.arange(1, num_products + 1)
    categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], size=num_products)
    base_prices = np.random.uniform(10, 500, size=num_products).round(2)
    
    products_df = pd.DataFrame({
        'product_id': product_ids,
        'category': categories,
        'base_price': base_prices
    })
    
    # 3. Generate Transactions
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    transactions = []
    
    # Simulate seasonal trends and daily noise
    for date in dates:
        # Determine daily volume (base + noise + weekend effect)
        is_weekend = date.weekday() >= 5
        base_volume = np.random.normal(loc=150, scale=30)
        daily_transactions = int(base_volume * (1.5 if is_weekend else 1.0))
        daily_transactions = max(50, daily_transactions)
        
        # Select random customers and products for the day
        daily_customers = np.random.choice(customer_ids, size=daily_transactions, replace=True)
        
        # Adjust product probability based on season (simple example)
        month = date.month
        
        daily_products = np.random.choice(product_ids, size=daily_transactions, replace=True)
        quantities = np.random.poisson(lam=1.5, size=daily_transactions)
        quantities = np.clip(quantities, 1, 10)
        
        for c_id, p_id, q in zip(daily_customers, daily_products, quantities):
            transactions.append({
                'transaction_id': len(transactions) + 1,
                'customer_id': c_id,
                'product_id': p_id,
                'date': date,
                'quantity': q
            })
            
    transactions_df = pd.DataFrame(transactions)
    
    # Merge price information
    transactions_df = transactions_df.merge(products_df[['product_id', 'base_price', 'category']], on='product_id')
    
    # Add random discounts
    discount_rates = np.random.choice([0, 0.1, 0.2, 0.3], size=len(transactions_df), p=[0.7, 0.15, 0.1, 0.05])
    transactions_df['discount'] = discount_rates
    transactions_df['total_amount'] = (transactions_df['quantity'] * transactions_df['base_price'] * (1 - transactions_df['discount'])).round(2)
    
    # Save to CSV
    os.makedirs('datasets', exist_ok=True)
    customers_df.to_csv('datasets/customers.csv', index=False)
    products_df.to_csv('datasets/products.csv', index=False)
    transactions_df.to_csv('datasets/transactions.csv', index=False)
    
    print(f"Generated {len(customers_df)} customers, {len(products_df)} products, and {len(transactions_df)} transactions.")
    print("Data saved to 'datasets/' directory.")

if __name__ == "__main__":
    generate_ecommerce_data()
