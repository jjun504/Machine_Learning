import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
import mindspore as ms
import mindspore.ops as ops
from src import config
from src.matrix_factorization import load_mf_model_and_mappings

def extract_features_and_candidates():
    print("Extracting features and generating candidates for XGBoost...")
    
    # 1. Load data files
    orders = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_products = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    train_products = pd.read_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH)
    # print("orders count: ", len(orders))
    
    prior_orders = orders[orders['eval_set'] == 'prior'].copy()
    
    # Merge prior products with orders for getting detailed info
    prior_details = pd.merge(prior_products, prior_orders, on='order_id')
    
    # 2. Calculate User Features
    print("Computing User features...")
    user_features = pd.DataFrame()
    # Total count of prior orders
    user_features['user_total_orders'] = prior_orders.groupby('user_id')['order_number'].max()
    # Average size of basket
    # # user_features['user_avg_basket_size'] = prior_details.groupby('user_id')['product_id'].count() / user_features['user_total_orders']
    basket_sizes = prior_details.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
    user_features['user_avg_basket_size'] = basket_sizes.groupby('user_id')['basket_size'].mean()
    # User reorder rate
    user_features['user_reorder_rate'] = prior_details.groupby('user_id')['reordered'].mean()
    # Average days since prior order
    user_features['user_avg_days_since_prior_order'] = prior_orders.groupby('user_id')['days_since_prior_order'].mean()
    user_features = user_features.reset_index()
    
    # 3. Product Features
    print("Computing Product features...")
    product_features = pd.DataFrame()
    # Total sales
    product_features['product_total_orders'] = prior_details.groupby('product_id').size()
    # Product reorder rate
    product_features['product_reorder_rate'] = prior_details.groupby('product_id')['reordered'].mean()
    # Average add to cart position
    product_features['product_avg_add_to_cart'] = prior_details.groupby('product_id')['add_to_cart_order'].mean()
    product_features = product_features.reset_index()
    
    # 4. User-Product Features
    print("Computing User-Product interaction features...")
    up_features = pd.DataFrame()
    # Total times purchased
    up_features['up_total_orders'] = prior_details.groupby(['user_id', 'product_id']).size()
    # Average add to cart position for user
    up_features['up_avg_add_to_cart'] = prior_details.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
    # Last order number this user purchased this product
    up_features['up_last_order_number'] = prior_details.groupby(['user_id', 'product_id'])['order_number'].max()
    up_features = up_features.reset_index()
    
    # Merge up features with user features to compute relative purchase rate and order distance
    up_features = pd.merge(up_features, user_features[['user_id', 'user_total_orders']], on='user_id')
    up_features['up_purchase_rate'] = up_features['up_total_orders'] / up_features['user_total_orders']
    up_features['up_last_order_distance'] = up_features['user_total_orders'] - up_features['up_last_order_number']
    # Drop temp column
    up_features = up_features.drop('user_total_orders', axis=1)
    
    # 5. Generate Candidates: For each user's target order, candidates are all products they bought in prior orders
    print("Generating candidate user-product pairs...")
    candidates = prior_details[['user_id', 'product_id']].drop_duplicates().copy()
    
    # Merge target order details to get split and target info
    target_orders = orders[orders['eval_set'] == 'train'][['order_id', 'user_id', 'split']].copy()
    candidates = pd.merge(candidates, target_orders, on='user_id')
    
    # Merge Features
    print("Merging features into candidates...")
    candidates = pd.merge(candidates, user_features, on='user_id', how='left')
    candidates = pd.merge(candidates, product_features, on='product_id', how='left')
    candidates = pd.merge(candidates, up_features, on=['user_id', 'product_id'], how='left')
    
    # 6. Generate Target (1 if product is in the target train order, 0 otherwise)
    print("Generating targets...")
    train_products['target'] = 1
    candidates = pd.merge(
        candidates, 
        train_products[['order_id', 'product_id', 'target']], 
        on=['order_id', 'product_id'], 
        how='left'
    )
    candidates['target'] = candidates['target'].fillna(0).astype(np.int32)
    
    # 7. Add Matrix Factorization Feature
    print("Adding Matrix Factorization interaction scores...")
    try:
        mf_model, mf_mappings = load_mf_model_and_mappings()
        user_to_idx = mf_mappings['user_to_idx']
        
        # Filter candidates to only those users mapped in MF model
        mapped_mask = candidates['user_id'].isin(user_to_idx)
        candidates_mapped = candidates[mapped_mask].copy()
        
        user_idxs = candidates_mapped['user_id'].map(user_to_idx).values.astype(np.int32)
        product_ids = candidates_mapped['product_id'].values.astype(np.int32)
        
        # Predict scores in batches to be fast and memory efficient in MindSpore PYNATIVE mode
        mf_scores = []
        batch_size = 4096
        num_candidates = len(user_idxs)
        
        for i in range(0, num_candidates, batch_size):
            u_b = ms.Tensor(user_idxs[i : i + batch_size])
            p_b = ms.Tensor(product_ids[i : i + batch_size])
            # Sigmoid(logits)
            logits = mf_model(u_b, p_b)
            probs = ops.sigmoid(logits).asnumpy()
            mf_scores.extend(probs.tolist())
            
        candidates_mapped['mf_score'] = mf_scores
        
        # Merge back
        candidates = pd.merge(
            candidates,
            candidates_mapped[['user_id', 'product_id', 'mf_score']],
            on=['user_id', 'product_id'],
            how='left'
        )
        # Fill missing with 0 (e.g. if any unmapped users, though should be none)
        candidates['mf_score'] = candidates['mf_score'].fillna(0.0)
        
    except Exception as e:
        print(f"Warning: Could not add MF feature due to error: {e}. Defaulting to 0.0.")
        candidates['mf_score'] = 0.0
        
    return candidates

def calculate_mean_f1(y_true_df, y_pred_probs, candidates_df, threshold=0.2):
    """
    Calculate the Mean F1-score across users for predicted baskets.
    """
    eval_df = candidates_df[['user_id', 'product_id', 'target']].copy()
    eval_df['pred_prob'] = y_pred_probs
    
    # Predict reordered if prob > threshold
    eval_df['pred_reordered'] = (eval_df['pred_prob'] >= threshold).astype(int)
    
    # Group actual and predicted products by user
    actual_baskets = eval_df[eval_df['target'] == 1].groupby('user_id')['product_id'].apply(set).to_dict()
    pred_baskets = eval_df[eval_df['pred_reordered'] == 1].groupby('user_id')['product_id'].apply(set).to_dict()
    
    # All evaluation users
    all_eval_users = eval_df['user_id'].unique()
    
    f1_scores = []
    for uid in all_eval_users:
        actual = actual_baskets.get(uid, set())
        pred = pred_baskets.get(uid, set())
        
        if len(actual) == 0 and len(pred) == 0:
            f1_scores.append(1.0)
            continue
        if len(actual) == 0 or len(pred) == 0:
            f1_scores.append(0.0)
            continue
            
        intersection = len(actual.intersection(pred))
        precision = intersection / len(pred)
        recall = intersection / len(actual)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
            
    return np.mean(f1_scores)

def train_xgboost():
    # 1. Extract candidates and features
    candidates = extract_features_and_candidates()
    
    # Define features to use
    feature_cols = [
        'user_total_orders', 'user_avg_basket_size', 'user_reorder_rate', 'user_avg_days_since_prior_order',
        'product_total_orders', 'product_reorder_rate', 'product_avg_add_to_cart',
        'up_total_orders', 'up_avg_add_to_cart', 'up_purchase_rate', 'up_last_order_distance',
        'mf_score'
    ]
    
    print(f"Features list: {feature_cols}")
    
    # 2. Split train and validation set based on split column
    train_data = candidates[candidates['split'] == 'train']
    val_data = candidates[candidates['split'] == 'val']
    
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    
    X_val = val_data[feature_cols].values
    y_val = val_data['target'].values
    
    print(f"XGBoost Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # 3. Train model
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        learning_rate=config.XGB_LEARNING_RATE,
        max_depth=config.XGB_MAX_DEPTH,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        random_state=config.RANDOM_SEED,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)
    
    # Save model
    model_path = os.path.join(config.PROCESSED_DATA_DIR, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, feature_cols), f)
        
    print(f"XGBoost model saved to {model_path}")
    
    # 4. Predict and evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    
    # ROC AUC
    roc_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation ROC AUC: {roc_auc:.4f}")
    
    # Sweep threshold to find best F1-score
    print("Optimizing classification threshold for Mean F1-Score...")
    best_f1 = 0
    best_thresh = 0.2
    
    for thresh in np.arange(0.1, 0.5, 0.05):
        f1 = calculate_mean_f1(y_val, val_preds, val_data, threshold=thresh)
        print(f"  Threshold {thresh:.2f} | Mean F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Best Validation Mean F1-Score: {best_f1:.4f} at threshold {best_thresh:.2f}")
    
    # Feature Importances
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  Feature {name:30s} | Importance: {imp:.4f}")
        
    # Save results
    metrics = {
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'best_threshold': best_thresh,
        'feature_importances': dict(zip(feature_cols, importances.tolist()))
    }
    with open(os.path.join(config.PROCESSED_DATA_DIR, "xgb_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
        
    return model, metrics

if __name__ == "__main__":
    train_xgboost()
