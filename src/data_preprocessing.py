import os
import pandas as pd
import numpy as np
from src import config

def run_preprocessing(force=False):
    # Check if files already exist
    if not force and os.path.exists(config.ORDERS_SAMPLE_PATH) and \
       os.path.exists(config.PRIOR_PRODUCTS_SAMPLE_PATH) and \
       os.path.exists(config.TRAIN_PRODUCTS_SAMPLE_PATH):
        print("Preprocessed sample files already exist. Skipping preprocessing.")
        return

    print("Starting data preprocessing and sampling...")
    np.random.seed(config.RANDOM_SEED)

    # 1. Load orders.csv
    orders_path = os.path.join(config.RAW_DATA_DIR, "orders.csv")
    print(f"Loading orders from {orders_path}...")
    orders = pd.read_csv(orders_path)
    
    # 2. Filter users with a train order
    train_users = orders[orders['eval_set'] == 'train']['user_id'].unique()
    print(f"Total users with a target order in train set: {len(train_users)}")
    
    # 3. Sample users
    sampled_users = np.random.choice(
        train_users, 
        size=min(config.NUM_USERS, len(train_users)), 
        replace=False
    )
    print(f"Sampled {len(sampled_users)} users for training and validation.")
    
    # 4. Split users into Train (80%) and Validation (20%)
    shuffled_users = sampled_users.copy()
    np.random.shuffle(shuffled_users)
    split_idx = int(len(shuffled_users) * 0.8)
    train_user_set = set(shuffled_users[:split_idx])
    
    # 5. Filter orders for sampled users
    orders_sample = orders[orders['user_id'].isin(sampled_users)].copy()
    
    # Add a split column to indicate whether the user belongs to train or validation split
    # Only the target order (eval_set == 'train') will be evaluated, but we mark the split at user-level
    orders_sample['split'] = orders_sample['user_id'].apply(
        lambda uid: 'train' if uid in train_user_set else 'val'
    )
    
    # 6. Load prior product orders
    prior_path = os.path.join(config.RAW_DATA_DIR, "order_products__prior.csv")
    print(f"Loading prior order products from {prior_path} (this may take a few seconds)...")
    prior_chunks = []
    # Using chunksize to load efficiently and keep memory footprint low
    sampled_order_ids = set(orders_sample['order_id'])
    
    for chunk in pd.read_csv(prior_path, chunksize=1000000):
        filtered_chunk = chunk[chunk['order_id'].isin(sampled_order_ids)]
        prior_chunks.append(filtered_chunk)
        
    prior_sample = pd.concat(prior_chunks, ignore_index=True)
    print(f"Filtered prior products sample: {len(prior_sample)} records.")
    
    # 7. Load train product orders
    train_prod_path = os.path.join(config.RAW_DATA_DIR, "order_products__train.csv")
    print(f"Loading train order products from {train_prod_path}...")
    train_products = pd.read_csv(train_prod_path)
    train_sample = train_products[train_products['order_id'].isin(sampled_order_ids)].copy()
    print(f"Filtered train products sample: {len(train_sample)} records.")
    
    # 8. Save samples
    print(f"Saving preprocessed samples to {config.PROCESSED_DATA_DIR}...")
    orders_sample.to_csv(config.ORDERS_SAMPLE_PATH, index=False)
    prior_sample.to_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH, index=False)
    train_sample.to_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing(force=True)
