import os
import time
import pickle
import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import run_preprocessing
from src.matrix_factorization import train_matrix_factorization, load_mf_model_and_mappings
from src.xgboost_model import train_xgboost, calculate_mean_f1
from src.transformer_model import train_transformer
import mindspore as ms
import mindspore.ops as ops
from sklearn.metrics import roc_auc_score

def evaluate_matrix_factorization_val():
    """
    Evaluate the trained Matrix Factorization model on the validation candidate set
    to provide a direct, fair comparison with XGBoost.
    """
    print("Evaluating Matrix Factorization model on validation candidate set...")
    
    # Load orders and validation split
    orders = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    train_products = pd.read_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH)
    prior_products = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    
    prior_orders = orders[orders['eval_set'] == 'prior'].copy()
    prior_details = pd.merge(prior_products, prior_orders, on='order_id')
    
    # Generate validation candidates
    target_orders = orders[orders['eval_set'] == 'train'][['order_id', 'user_id', 'split']].copy()
    val_orders = target_orders[target_orders['split'] == 'val']
    
    candidates = prior_details[['user_id', 'product_id']].drop_duplicates().copy()
    candidates = pd.merge(candidates, val_orders, on='user_id')
    
    # Targets
    train_products['target'] = 1
    candidates = pd.merge(
        candidates, 
        train_products[['order_id', 'product_id', 'target']], 
        on=['order_id', 'product_id'], 
        how='left'
    )
    candidates['target'] = candidates['target'].fillna(0).astype(np.int32)
    
    # Load model and mappings
    model, mappings = load_mf_model_and_mappings()
    user_to_idx = mappings['user_to_idx']
    num_products_vocab = mappings['num_products_vocab']
    
    # Predict for mapped candidates
    mapped_mask = candidates['user_id'].isin(user_to_idx)
    candidates_mapped = candidates[mapped_mask].copy()
    
    user_idxs = candidates_mapped['user_id'].map(user_to_idx).values.astype(np.int32)
    product_ids = candidates_mapped['product_id'].values.astype(np.int32)
    
    mf_scores = []
    batch_size = 4096
    num_candidates = len(user_idxs)
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    for i in range(0, num_candidates, batch_size):
        u_b = ms.Tensor(user_idxs[i : i + batch_size])
        p_b = ms.Tensor(product_ids[i : i + batch_size])
        logits = model(u_b, p_b)
        probs = ops.sigmoid(logits).asnumpy()
        mf_scores.extend(probs.tolist())
        
    candidates_mapped['mf_score'] = mf_scores
    
    y_val = candidates_mapped['target'].values
    val_preds = candidates_mapped['mf_score'].values
    
    # ROC AUC
    roc_auc = roc_auc_score(y_val, val_preds)
    
    # Sweep threshold for Mean F1
    # for thresh in [0.1, 0.2, 0.3, 0.4]:
    #     f1 = calculate_mean_f1(y_val, val_preds, candidates_mapped, threshold=thresh)
    #     print(f"thresh {thresh} f1 {f1}")
    best_f1 = 0.0
    best_thresh = 0.2
    for thresh in np.arange(0.1, 0.5, 0.05):
        f1 = calculate_mean_f1(y_val, val_preds, candidates_mapped, threshold=thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"MF Validation ROC AUC: {roc_auc:.4f} | Best F1-Score: {best_f1:.4f} at threshold {best_thresh:.2f}")
    return roc_auc, best_f1

def evaluate_transformer_candidate_val():
    """
    Evaluate the trained Transformer model on the validation candidate set
    to provide a 100% fair apples-to-apples comparison.
    """
    print("Evaluating Transformer model on validation candidate set...")
    
    # Load preprocessed files
    orders_sample = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_sample = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    train_sample = pd.read_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH)
    
    from src.transformer_model import prepare_transformer_sequences, SequentialTransformer
    dataset = prepare_transformer_sequences(orders_sample, prior_sample, train_sample)
    
    # Load mappings
    mappings_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
    num_products_vocab = mappings['num_products_vocab']
    
    # Load model
    model = SequentialTransformer(
        num_products=num_products_vocab,
        embedding_dim=config.TRANSFORMER_EMBEDDING_DIM,
        num_heads=config.TRANSFORMER_NUM_HEADS,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN
    )
    
    ckpt_path = os.path.join(config.PROCESSED_DATA_DIR, "transformer_model.ckpt")
    ms.load_checkpoint(ckpt_path, model)
    model.set_train(False)
    
    splits = np.array(dataset['splits'])
    val_mask = (splits == 'val')
    
    seqs_val = dataset['sequences'][val_mask]
    masks_val = dataset['masks'][val_mask]
    
    num_val = len(seqs_val)
    batch_size = config.TRANSFORMER_BATCH_SIZE
    val_preds_vocab = []
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    for i in range(0, num_val, batch_size):
        s_b = ms.Tensor(seqs_val[i : i + batch_size], dtype=ms.int32)
        m_b = ms.Tensor(masks_val[i : i + batch_size], dtype=ms.int32)
        logits = model(s_b, m_b)
        probs = ops.sigmoid(logits).asnumpy()
        val_preds_vocab.append(probs)
        
    val_preds_vocab = np.concatenate(val_preds_vocab, axis=0)
    # print("val_preds_vocab shape:", val_preds_vocab.shape)
    
    # Generate candidates
    prior_orders = orders_sample[orders_sample['eval_set'] == 'prior'].copy()
    prior_details = pd.merge(prior_sample, prior_orders, on='order_id')
    
    target_orders = orders_sample[orders_sample['eval_set'] == 'train'][['order_id', 'user_id', 'split']].copy()
    val_orders = target_orders[target_orders['split'] == 'val']
    
    candidates = prior_details[['user_id', 'product_id']].drop_duplicates().copy()
    candidates = pd.merge(candidates, val_orders, on='user_id')
    
    train_sample['target'] = 1
    candidates = pd.merge(
        candidates, 
        train_sample[['order_id', 'product_id', 'target']], 
        on=['order_id', 'product_id'], 
        how='left'
    )
    candidates['target'] = candidates['target'].fillna(0).astype(np.int32)
    
    # Map predictions
    val_users = orders_sample[(orders_sample['eval_set'] == 'train') & (orders_sample['split'] == 'val')]['user_id'].values
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(val_users)}
    
    user_ids = candidates['user_id'].values
    prod_ids = candidates['product_id'].values
    
    tf_scores = []
    for user_id, prod_id in zip(user_ids, prod_ids):
        user_offset = user_id_to_idx[user_id]
        if prod_id < num_products_vocab:
            prob = val_preds_vocab[user_offset, int(prod_id)]
        else:
            prob = 0.0
        tf_scores.append(prob)
        
    candidates['tf_score'] = tf_scores
    
    y_val = candidates['target'].values
    val_preds = candidates['tf_score'].values
    
    roc_auc = roc_auc_score(y_val, val_preds)
    
    best_f1 = 0.0
    best_thresh = 0.1
    for thresh in np.arange(0.1, 0.5, 0.05):
        f1 = calculate_mean_f1(y_val, val_preds, candidates, threshold=thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"TF Candidate-restricted Validation ROC AUC: {roc_auc:.4f} | Best F1-Score: {best_f1:.4f} at threshold {best_thresh:.2f}")
    return roc_auc, best_f1

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
    # print("MF Time elapsed:", times['Matrix Factorization'])
    print("-" * 50)
    
    # Step 3: XGBoost Model
    t_start = time.time()
    train_xgboost()
    times['XGBoost'] = time.time() - t_start
    # print("XGB Time elapsed:", times['XGBoost'])
    print("-" * 50)
    
    # Step 4: Transformer Model
    t_start = time.time()
    train_transformer()
    times['Transformer'] = time.time() - t_start
    # print("TF Time elapsed:", times['Transformer'])
    print("-" * 50)
    
    # Load saved metrics
    with open(os.path.join(config.PROCESSED_DATA_DIR, "xgb_metrics.pkl"), "rb") as f:
        xgb_metrics = pickle.load(f)
    with open(os.path.join(config.PROCESSED_DATA_DIR, "transformer_metrics.pkl"), "rb") as f:
        tf_metrics = pickle.load(f)
        
    # Evaluate MF model on validation set
    mf_auc, mf_f1 = evaluate_matrix_factorization_val()
    
    # Evaluate Transformer on validation candidate set for fair comparison
    tf_cand_auc, tf_cand_f1 = evaluate_transformer_candidate_val()
    
    # Display comparison table
    print("\n" + "="*60)
    print("               ML Pipeline Comparison Results")
    print("="*60)
    
    results = [
        {
            'Model': 'Matrix Factorization (NCF)',
            'Validation ROC AUC': f"{mf_auc:.4f}",
            'Mean F1-Score': f"{mf_f1:.4f}",
            'Training Time': f"{times['Matrix Factorization']/60:.2f} min"
        },
        {
            'Model': 'XGBoost (Classifier)',
            'Validation ROC AUC': f"{xgb_metrics['roc_auc']:.4f}",
            'Mean F1-Score': f"{xgb_metrics['best_f1']:.4f}",
            'Training Time': f"{times['XGBoost']/60:.2f} min"
        },
        {
            'Model': 'Sequential Transformer (Encoder - Fair)',
            'Validation ROC AUC': f"{tf_cand_auc:.4f}",
            'Mean F1-Score': f"{tf_cand_f1:.4f}",
            'Training Time': f"{times['Transformer']/60:.2f} min"
        },
        {
            'Model': 'Sequential Transformer (Encoder - Full Vocab)',
            'Validation ROC AUC': f"{tf_metrics['roc_auc']:.4f}",
            'Mean F1-Score': f"{tf_metrics['f1_score']:.4f} (Top-10)",
            'Training Time': f"{times['Transformer']/60:.2f} min"
        }
    ]
    
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    print("="*60)
    
    # Save comparison results
    with open(os.path.join(config.PROCESSED_DATA_DIR, "comparison_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Comparison results saved.")

if __name__ == "__main__":
    main()
