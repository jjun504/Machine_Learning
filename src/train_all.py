import os
import time
import pickle
import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import run_preprocessing
from src.matrix_factorization import train_matrix_factorization, load_mf_model_and_mappings
from src.xgboost_model import train_xgboost, extract_features_and_candidates, calculate_mean_f1
from src.transformer_model import train_transformer, prepare_transformer_sequences, SequentialTransformer
import mindspore as ms
import mindspore.ops as ops
from sklearn.metrics import roc_auc_score

def compute_rec_metrics(y_true, scores, candidates_df, K=10):
    """
    Compute Precision@K, Recall@K, Hit Rate@K, NDCG@K, MRR, and ROC AUC
    for a given set of predictions.
    """
    candidates_df = candidates_df.copy()
    candidates_df['score'] = scores
    candidates_df['target'] = y_true
    
    user_precisions = []
    user_recalls = []
    user_hits = []
    user_ndcgs = []
    user_mrrs = []
    
    grouped = candidates_df.groupby('user_id')
    for user_id, group in grouped:
        actual_set = set(group[group['target'] == 1]['product_id'].values)
        if len(actual_set) == 0:
            continue
            
        sorted_group = group.sort_values(by='score', ascending=False)
        top_k_preds = sorted_group['product_id'].values[:K]
        
        # 1. Precision@K
        hits = len(set(top_k_preds).intersection(actual_set))
        precision = hits / K
        user_precisions.append(precision)
        
        # 2. Recall@K
        recall = hits / len(actual_set)
        user_recalls.append(recall)
        
        # 3. Hit Rate@K
        hit = 1.0 if hits > 0 else 0.0
        user_hits.append(hit)
        
        # 4. NDCG@K
        dcg = 0.0
        for idx, item in enumerate(top_k_preds):
            if item in actual_set:
                dcg += 1.0 / np.log2(idx + 2)
        
        idcg = 0.0
        for idx in range(min(K, len(actual_set))):
            idcg += 1.0 / np.log2(idx + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        user_ndcgs.append(ndcg)
        
        # 5. MRR
        mrr = 0.0
        for idx, item in enumerate(top_k_preds):
            if item in actual_set:
                mrr = 1.0 / (idx + 1)
                break
        user_mrrs.append(mrr)
        
    mean_precision = np.mean(user_precisions)
    mean_recall = np.mean(user_recalls)
    mean_hit_rate = np.mean(user_hits)
    mean_ndcg = np.mean(user_ndcgs)
    mean_mrr = np.mean(user_mrrs)
    
    roc_auc = roc_auc_score(y_true, scores)
    
    if mean_precision + mean_recall > 0:
        f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    else:
        f1 = 0.0
        
    return {
        'ROC AUC': roc_auc,
        'F1@K': f1,
        'Precision@K': mean_precision,
        'Recall@K': mean_recall,
        'Hit Rate@K': mean_hit_rate,
        'NDCG@K': mean_ndcg,
        'MRR': mean_mrr
    }

def evaluate_all_models(times=None):
    """
    Runs the comprehensive metrics calculation for all three models
    and returns a pandas DataFrame with the results.
    """
    print("Running comprehensive recommendation metrics benchmarking...")
    
    # Load all candidates with their features
    candidates = extract_features_and_candidates()
    
    # Filter validation set
    val_mask = (candidates['split'] == 'val')
    candidates_val = candidates[val_mask].copy()
    y_val = candidates_val['target'].values
    
    # 1. NCF Evaluation
    print("Evaluating Matrix Factorization (NCF)...")
    mf_scores = candidates_val['mf_score'].values
    mf_metrics = compute_rec_metrics(y_val, mf_scores, candidates_val)
    
    # 2. XGBoost Evaluation
    print("Evaluating XGBoost...")
    with open(os.path.join(config.PROCESSED_DATA_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model, feature_cols = pickle.load(f)
    X_val = candidates_val[feature_cols].values
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_metrics = compute_rec_metrics(y_val, xgb_probs, candidates_val)
    
    # 3. Transformer Evaluation (Candidate-restricted)
    print("Evaluating Transformer (Fair Comparison on Candidates)...")
    orders_sample = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_sample = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    train_sample = pd.read_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH)
    
    dataset = prepare_transformer_sequences(orders_sample, prior_sample, train_sample)
    
    mappings_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
    num_products_vocab = mappings['num_products_vocab']
    
    tf_model = SequentialTransformer(
        num_products=num_products_vocab,
        embedding_dim=config.TRANSFORMER_EMBEDDING_DIM,
        num_heads=config.TRANSFORMER_NUM_HEADS,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN
    )
    ckpt_path = os.path.join(config.PROCESSED_DATA_DIR, "transformer_model.ckpt")
    ms.load_checkpoint(ckpt_path, tf_model)
    tf_model.set_train(False)
    
    splits = np.array(dataset['splits'])
    val_mask_seq = (splits == 'val')
    seqs_val = dataset['sequences'][val_mask_seq]
    masks_val = dataset['masks'][val_mask_seq]
    
    val_preds_vocab = []
    batch_size = config.TRANSFORMER_BATCH_SIZE
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    for i in range(0, len(seqs_val), batch_size):
        s_b = ms.Tensor(seqs_val[i : i + batch_size], dtype=ms.int32)
        m_b = ms.Tensor(masks_val[i : i + batch_size], dtype=ms.int32)
        logits = tf_model(s_b, m_b)
        probs = ops.sigmoid(logits).asnumpy()
        val_preds_vocab.append(probs)
    val_preds_vocab = np.concatenate(val_preds_vocab, axis=0)
    
    val_users = orders_sample[(orders_sample['eval_set'] == 'train') & (orders_sample['split'] == 'val')]['user_id'].values
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(val_users)}
    
    user_ids = candidates_val['user_id'].values
    prod_ids = candidates_val['product_id'].values
    
    tf_scores = []
    for user_id, prod_id in zip(user_ids, prod_ids):
        user_offset = user_id_to_idx[user_id]
        if prod_id < num_products_vocab:
            prob = val_preds_vocab[user_offset, int(prod_id)]
        else:
            prob = 0.0
        tf_scores.append(prob)
        
    candidates_val['tf_score'] = tf_scores
    tf_metrics = compute_rec_metrics(y_val, candidates_val['tf_score'].values, candidates_val)
    
    # ----------------------------------------------------
    # Compile Results DataFrame
    # ----------------------------------------------------
    results = [
        {**{'Model': 'Matrix Factorization (NCF)'}, **mf_metrics, **{'Training Time': f"{times['Matrix Factorization']/60:.2f} min" if times else "N/A"}},
        {**{'Model': 'XGBoost (Classifier)'}, **xgb_metrics, **{'Training Time': f"{times['XGBoost']/60:.2f} min" if times else "N/A"}},
        {**{'Model': 'Sequential Transformer'}, **tf_metrics, **{'Training Time': f"{times['Transformer']/60:.2f} min" if times else "N/A"}}
    ]
    
    df_results = pd.DataFrame(results)
    
    # Save comparison results
    with open(os.path.join(config.PROCESSED_DATA_DIR, "comprehensive_metrics.pkl"), "wb") as f:
        pickle.dump(results, f)
        
    return df_results

def main():
    print("="*60)
    print("Instacart Market Basket Analysis ML Pipeline Orchestrator")
    print("="*60)
    
    times = {}
    
    # Step 1: Preprocessing
    t_start = time.time()
    run_preprocessing(force=False)
    times['Preprocessing'] = time.time() - t_start
    print("-" * 50)
    
    # Step 2: Matrix Factorization
    t_start = time.time()
    train_matrix_factorization()
    times['Matrix Factorization'] = time.time() - t_start
    print("-" * 50)
    
    # Step 3: XGBoost Model
    t_start = time.time()
    train_xgboost()
    times['XGBoost'] = time.time() - t_start
    print("-" * 50)
    
    # Step 4: Transformer Model
    t_start = time.time()
    train_transformer()
    times['Transformer'] = time.time() - t_start
    print("-" * 50)
    
    # Evaluate
    df_results = evaluate_all_models(times)
    
    print("\n" + "="*80)
    print("               ML Pipeline Benchmarking Comparison Results")
    print("="*80)
    print(df_results.to_markdown(index=False))
    print("="*80)
    print("Comprehensive metrics saved.")

if __name__ == "__main__":
    main()
