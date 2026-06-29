import os
import pickle
import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from src import config

# Set MindSpore context
import warnings
warnings.filterwarnings("ignore")
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class MatrixFactorization(nn.Cell):
    """
    Neural Collaborative Filtering (NCF) / Matrix Factorization model in MindSpore.
    Computes: prediction = sigmoid(User_Embedding . Product_Embedding + User_Bias + Product_Bias)
    """
    def __init__(self, num_users, num_products, embedding_dim=64):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim, embedding_table="xavier_uniform")
        self.prod_emb = nn.Embedding(num_products, embedding_dim, embedding_table="xavier_uniform")
        self.user_bias = nn.Embedding(num_users, 1, embedding_table="zeros")
        self.prod_bias = nn.Embedding(num_products, 1, embedding_table="zeros")

    def construct(self, user_ids, product_ids):
        # Squeeze inputs to handle potential 2D tensors from dataset pipeline
        u_ids = ops.squeeze(user_ids)
        p_ids = ops.squeeze(product_ids)
        
        # Get embeddings
        u_e = self.user_emb(u_ids)  # Shape: (batch_size, embedding_dim)
        p_e = self.prod_emb(p_ids)  # Shape: (batch_size, embedding_dim)
        
        # Dot product
        dot = ops.reduce_sum(u_e * p_e, axis=1)  # Shape: (batch_size,)
        
        # Biases
        u_b = ops.squeeze(self.user_bias(u_ids))  # Shape: (batch_size,)
        p_b = ops.squeeze(self.prod_bias(p_ids))  # Shape: (batch_size,)
        
        # Combine
        logits = dot + u_b + p_b
        return logits

def prepare_mf_data(orders_sample, prior_sample):
    """
    Prepare implicit feedback training data with negative sampling.
    """
    print("Preparing data for Matrix Factorization...")
    
    # 1. Get unique users and products in the sample
    unique_users = orders_sample['user_id'].unique()
    unique_products = prior_sample['product_id'].unique()
    
    # Create user index mapping (to map large user_ids to 0..N-1)
    user_to_idx = {uid: idx for idx, uid in enumerate(sorted(unique_users))}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    
    # Product index mapping (can map product_ids to 0..M-1, or use them directly if size allows)
    # Using 1-indexed product IDs directly (embedding size = max_prod_id + 1)
    max_product_id = int(prior_sample['product_id'].max())
    num_products_vocab = max_product_id + 1
    
    # 2. Get positive interactions (users and products they bought)
    # Group by user_id and product_id from orders and order_products__prior
    prior_orders = orders_sample[orders_sample['eval_set'] == 'prior']
    merged = pd.merge(prior_sample, prior_orders, on='order_id')
    user_product_pairs = merged[['user_id', 'product_id']].drop_duplicates()
    
    # Map user IDs
    user_product_pairs['user_idx'] = user_product_pairs['user_id'].map(user_to_idx)
    
    pos_users = user_product_pairs['user_idx'].values
    pos_products = user_product_pairs['product_id'].values
    
    num_positives = len(pos_users)
    # print("Positive users count:", len(pos_users))
    print(f"Positive interaction pairs: {num_positives}")
    
    # 3. Do negative sampling: for each positive, sample 4 negatives
    num_negatives = num_positives * 4
    neg_users = np.repeat(pos_users, 4)
    
    # Group products by user to check for membership
    user_bought_set = user_product_pairs.groupby('user_idx')['product_id'].apply(set).to_dict()
    
    print("Sampling negative user-product pairs...")
    # neg_products = []
    # for u in neg_users:
    #     while True:
    #         p = np.random.choice(unique_products)
    #         if p not in user_bought_set[u]:
    #             neg_products.append(p)
    #             break
    
    # Pre-sample all negative products at once to optimize performance
    neg_products = np.random.choice(unique_products, size=len(neg_users))
    
    # Correct any collisions where the sampled product was actually purchased by the user
    collision_count = 0
    for i in range(len(neg_users)):
        u_idx = neg_users[i]
        p_id = neg_products[i]
        if p_id in user_bought_set[u_idx]:
            collision_count += 1
            while True:
                new_p = np.random.choice(unique_products)
                if new_p not in user_bought_set[u_idx]:
                    neg_products[i] = new_p
                    break
                    
    print(f"Negative sampling completed. Corrected {collision_count} collisions.")
    neg_products = np.array(neg_products, dtype=np.int32)
    
    # Combine positive and negative samples
    users_all = np.concatenate([pos_users, neg_users]).astype(np.int32)
    products_all = np.concatenate([pos_products, neg_products]).astype(np.int32)
    labels_all = np.concatenate([np.ones(num_positives), np.zeros(num_negatives)]).astype(np.float32)
    
    # Shuffle dataset
    shuffle_indices = np.random.permutation(len(users_all))
    users_all = users_all[shuffle_indices]
    products_all = products_all[shuffle_indices]
    labels_all = labels_all[shuffle_indices]
    
    # Save mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'num_users': len(unique_users),
        'num_products_vocab': num_products_vocab
    }
    
    with open(os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
        
    return users_all, products_all, labels_all, mappings

def train_matrix_factorization():
    orders_sample = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_sample = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    
    users_all, products_all, labels_all, mappings = prepare_mf_data(orders_sample, prior_sample)
    
    # Define model, loss, and optimizer
    model = MatrixFactorization(mappings['num_users'], mappings['num_products_vocab'], config.MF_EMBEDDING_DIM)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.MF_LR)
    
    # Define functional training loop step (following Lab8a2.ipynb style)
    def forward_fn(u, p, y):
        logits = model(u, p)
        loss = loss_fn(logits, y)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(u, p, y):
        loss, grads = grad_fn(u, p, y)
        optimizer(grads)
        return loss

    print("Training MindSpore Matrix Factorization model...", flush=True)
    model.set_train()
    
    num_samples = len(users_all)
    batch_size = config.MF_BATCH_SIZE
    
    epoch_losses = []
    for epoch in range(config.MF_EPOCHS):
        # Shuffle at the beginning of each epoch
        indices = np.random.permutation(num_samples)
        users_shuffled = users_all[indices]
        products_shuffled = products_all[indices]
        labels_shuffled = labels_all[indices]
        
        epoch_loss = 0.0
        steps = 0
        
        for i in range(0, num_samples, batch_size):
            u_b = ms.Tensor(users_shuffled[i : i + batch_size], dtype=ms.int32)
            p_b = ms.Tensor(products_shuffled[i : i + batch_size], dtype=ms.int32)
            y_b = ms.Tensor(labels_shuffled[i : i + batch_size], dtype=ms.float32)
            
            loss = train_step(u_b, p_b, y_b)
            epoch_loss += float(loss.asnumpy())
            steps += 1
            
        avg_loss = epoch_loss / steps
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{config.MF_EPOCHS}: Loss = {avg_loss:.4f}", flush=True)
        
    # Save checkpoint of model
    ckpt_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_model.ckpt")
    ms.save_checkpoint(model, ckpt_path)
    print(f"Matrix Factorization model save to {ckpt_path}", flush=True)
    
    # Save loss history in mappings
    mappings['loss_history'] = epoch_losses
    with open(os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
        
    return model, mappings

def grid_search_mf(param_grid=None):
    """
    Performs grid search over NCF hyperparameters on a small sample of users
    for speed, and prints the validation metrics for each combination.
    """
    if param_grid is None:
        param_grid = {
            'embedding_dim': [32, 64],
            'lr': [0.005, 0.01]
        }
    
    print("Preparing data for NCF Grid Search...", flush=True)
    orders_sample = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_sample = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    
    users_all, products_all, labels_all, mappings = prepare_mf_data(orders_sample, prior_sample)
    
    # Subsample for faster search
    np.random.seed(config.RANDOM_SEED)
    subset_indices = np.random.choice(len(users_all), size=min(10000, len(users_all)), replace=False)
    u_sub = users_all[subset_indices]
    p_sub = products_all[subset_indices]
    y_sub = labels_all[subset_indices]
    
    split = int(len(u_sub) * 0.8)
    u_train, u_val = u_sub[:split], u_sub[split:]
    p_train, p_val = p_sub[:split], p_sub[split:]
    y_train, y_val = y_sub[:split], y_sub[split:]
    
    best_loss = float('inf')
    best_params = {}
    
    print("NCF Grid Search started...", flush=True)
    for dim in param_grid['embedding_dim']:
        for lr in param_grid['lr']:
            model = MatrixFactorization(mappings['num_users'], mappings['num_products_vocab'], dim)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
            
            def forward_fn(u, p, y):
                return loss_fn(model(u, p), y)
            
            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            
            model.set_train()
            batch_size = 512
            # Train for 1 epoch
            for i in range(0, len(u_train), batch_size):
                u_b = ms.Tensor(u_train[i:i+batch_size], dtype=ms.int32)
                p_b = ms.Tensor(p_train[i:i+batch_size], dtype=ms.int32)
                y_b = ms.Tensor(y_train[i:i+batch_size], dtype=ms.float32)
                loss, grads = grad_fn(u_b, p_b, y_b)
                optimizer(grads)
            
            # Eval
            model.set_train(False)
            val_loss = 0.0
            val_steps = 0
            for i in range(0, len(u_val), batch_size):
                u_b = ms.Tensor(u_val[i:i+batch_size], dtype=ms.int32)
                p_b = ms.Tensor(p_val[i:i+batch_size], dtype=ms.int32)
                y_b = ms.Tensor(y_val[i:i+batch_size], dtype=ms.float32)
                loss = loss_fn(model(u_b, p_b), y_b)
                val_loss += float(loss.asnumpy())
                val_steps += 1
                
            avg_val_loss = val_loss / val_steps
            print(f"NCF Grid Search trial: embedding_dim={dim}, lr={lr} -> Val Loss={avg_val_loss:.4f}", flush=True)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'embedding_dim': dim, 'lr': lr}
                
    print(f"NCF Grid Search finished. Best Params: {best_params} (Val Loss: {best_loss:.4f})", flush=True)
    return best_params

def load_mf_model_and_mappings():
    """
    Helper function for load model and mappings
    """
    mappings_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl")
    ckpt_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_model.ckpt")
    
    if not os.path.exists(mappings_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Trained MF model or mappings not found. Please train it first.")
        
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    model = MatrixFactorization(mappings['num_users'], mappings['num_products_vocab'], config.MF_EMBEDDING_DIM)
    ms.load_checkpoint(ckpt_path, model)
    model.set_train(False)
    return model, mappings

if __name__ == "__main__":
    train_matrix_factorization()
