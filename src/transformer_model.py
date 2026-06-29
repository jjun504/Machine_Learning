import os
import pickle
import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from sklearn.metrics import roc_auc_score
from src import config

# Set MindSpore context
import warnings
warnings.filterwarnings("ignore")
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class SequentialTransformer(nn.Cell):
    """
    Order-level Sequential Transformer model in MindSpore.
    Aggregates product embeddings to represent orders, applies a Transformer Encoder,
    and projects the final user state to predict product probabilities in the next order.
    """
    def __init__(self, num_products, embedding_dim=64, num_heads=4, num_layers=2, max_seq_len=10):
        super(SequentialTransformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Shared product embedding
        self.prod_emb = nn.Embedding(num_products, embedding_dim, embedding_table="xavier_uniform")
        
        # Learned positional embedding for order sequence
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim, embedding_table="xavier_uniform")
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output classification layer (projects to all products)
        self.fc = nn.Dense(embedding_dim, num_products, weight_init="xavier_uniform")
        
    def construct(self, order_sequences, order_mask):
        # order_sequences shape: (batch_size, max_seq_len, max_products_per_order)
        # order_mask shape: (batch_size, max_seq_len)
        batch_size, seq_len, max_prod = order_sequences.shape
        
        # Flatten sequence to embed all products in one call
        flat_seqs = order_sequences.view(-1, max_prod)
        prod_embs = self.prod_emb(flat_seqs)  # (batch_size * seq_len, max_prod, embedding_dim)
        
        # Average product embeddings to get order representation (ignoring padding index 0)
        prod_mask = ops.expand_dims(flat_seqs > 0, -1).astype(ms.float32)
        prod_embs = prod_embs * prod_mask
        sum_embs = ops.reduce_sum(prod_embs, axis=1)
        cnt = ops.reduce_sum(prod_mask, axis=1)
        cnt = ops.maximum(cnt, 1.0)
        order_embs = sum_embs / cnt
        
        # Reshape to sequence format: (batch_size, seq_len, embedding_dim)
        order_embs = order_embs.view(batch_size, seq_len, -1)
        
        # Add Positional Embeddings
        pos_indices = ops.arange(seq_len, dtype=ms.int32)
        pos_embs = self.pos_emb(pos_indices)  # (seq_len, embedding_dim)
        x = order_embs + ops.expand_dims(pos_embs, 0)
        
        # Apply Transformer Encoder
        # MindSpore's padding mask should be boolean with True for positions to mask out
        padding_mask = (order_mask == 0)
        features = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Take the representation of the last order in the sequence
        user_state = features[:, -1, :]  # (batch_size, embedding_dim)
        
        # Project to vocabulary logits
        logits = self.fc(user_state)
        return logits

def prepare_transformer_sequences(orders_sample, prior_sample, train_sample):
    """
    Prepare order sequence history for each user.
    """
    print("Preparing order sequence data for Transformer...")
    
    # 1. Group prior products by order_id
    prior_orders = prior_sample.groupby('order_id')['product_id'].apply(list).to_dict()
    
    # 2. Group train products by order_id (target orders)
    train_orders = train_sample.groupby('order_id')['product_id'].apply(list).to_dict()
    
    # Load mappings from MF to keep vocabulary size identical
    mappings_path = os.path.join(config.PROCESSED_DATA_DIR, "mf_mappings.pkl")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    num_products_vocab = mappings['num_products_vocab']
    
    # 3. Build sequence history for each user
    user_sequences = []
    user_masks = []
    user_targets = []
    user_splits = []
    user_ids_list = []
    
    # Filter target orders
    target_orders_df = orders_sample[orders_sample['eval_set'] == 'train']
    
    # Prior orders grouped by user
    prior_user_orders = orders_sample[orders_sample['eval_set'] == 'prior'].sort_values(by='order_number')
    user_prior_orders_dict = prior_user_orders.groupby('user_id')['order_id'].apply(list).to_dict()
    
    max_seq_len = config.MAX_SEQ_LEN
    max_prod = config.MAX_PRODUCTS_PER_ORDER
    
    for _, row in target_orders_df.iterrows():
        uid = row['user_id']
        target_order_id = row['order_id']
        split = row['split']
        
        # Get prior order IDs
        u_orders = user_prior_orders_dict.get(uid, [])
        # Take last N orders
        u_orders_seq = u_orders[-max_seq_len:]
        
        # Prepare padded sequence of order vectors
        seq_vector = np.zeros((max_seq_len, max_prod), dtype=np.int32)
        mask_vector = np.zeros(max_seq_len, dtype=np.int32)
        
        # Fill sequence from the right (right-aligned, left-padded)
        start_idx = max_seq_len - len(u_orders_seq)
        for i, oid in enumerate(u_orders_seq):
            products = prior_orders.get(oid, [])
            # Truncate or pad products
            products_padded = products[:max_prod] + [0] * max(0, max_prod - len(products))
            seq_vector[start_idx + i] = products_padded
            mask_vector[start_idx + i] = 1
            
        # Target products in next order
        target_prods = train_orders.get(target_order_id, [])
        
        user_sequences.append(seq_vector)
        user_masks.append(mask_vector)
        user_targets.append(target_prods)
        user_splits.append(split)
        user_ids_list.append(uid)
        
    user_sequences = np.array(user_sequences, dtype=np.int32)
    user_masks = np.array(user_masks, dtype=np.int32)
    
    # Return formatted dataset
    dataset = {
        'sequences': user_sequences,
        'masks': user_masks,
        'targets': user_targets,
        'splits': user_splits,
        'user_ids': user_ids_list,
        'num_products_vocab': num_products_vocab
    }
    return dataset

def train_transformer():
    orders_sample = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    prior_sample = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
    train_sample = pd.read_csv(config.TRAIN_PRODUCTS_SAMPLE_PATH)
    
    dataset = prepare_transformer_sequences(orders_sample, prior_sample, train_sample)
    
    splits = np.array(dataset['splits'])
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    
    # Split training arrays
    seqs_train = dataset['sequences'][train_mask]
    masks_train = dataset['masks'][train_mask]
    targets_train = [dataset['targets'][i] for i in range(len(splits)) if train_mask[i]]
    
    seqs_val = dataset['sequences'][val_mask]
    masks_val = dataset['masks'][val_mask]
    targets_val = [dataset['targets'][i] for i in range(len(splits)) if val_mask[i]]
    
    num_products_vocab = dataset['num_products_vocab']
    
    # Define model, loss, and optimizer
    model = SequentialTransformer(
        num_products=num_products_vocab,
        embedding_dim=config.TRANSFORMER_EMBEDDING_DIM,
        num_heads=config.TRANSFORMER_NUM_HEADS,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config.TRANSFORMER_LR)
    
    def forward_fn(seqs, masks, targets):
        logits = model(seqs, masks)
        loss = loss_fn(logits, targets)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(seqs, masks, targets):
        loss, grads = grad_fn(seqs, masks, targets)
        optimizer(grads)
        return loss

    print("Training MindSpore Sequential Transformer model...", flush=True)
    model.set_train()
    
    num_train = len(seqs_train)
    batch_size = config.TRANSFORMER_BATCH_SIZE
    
    for epoch in range(config.TRANSFORMER_EPOCHS):
        # Shuffle train data
        indices = np.random.permutation(num_train)
        seqs_shuffled = seqs_train[indices]
        masks_shuffled = masks_train[indices]
        targets_shuffled = [targets_train[i] for i in indices]
        
        epoch_loss = 0.0
        steps = 0
        
        for i in range(0, num_train, batch_size):
            # Batch slicing
            s_b = ms.Tensor(seqs_shuffled[i : i + batch_size], dtype=ms.int32)
            m_b = ms.Tensor(masks_shuffled[i : i + batch_size], dtype=ms.int32)
            
            # Construct multi-hot target label vector on-the-fly to save memory
            cur_batch_size = len(s_b)
            targets_b_np = np.zeros((cur_batch_size, num_products_vocab), dtype=np.float32)
            for j in range(cur_batch_size):
                p_ids = targets_shuffled[i + j]
                # Filter valid product IDs
                p_ids_valid = [p for p in p_ids if p < num_products_vocab]
                targets_b_np[j, p_ids_valid] = 1.0
                
            t_b = ms.Tensor(targets_b_np, dtype=ms.float32)
            
            loss = train_step(s_b, m_b, t_b)
            epoch_loss += float(loss.asnumpy())
            steps += 1
            
        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch + 1}/{config.TRANSFORMER_EPOCHS}: Loss = {avg_loss:.4f}", flush=True)
        
    # Save checkpoint
    ckpt_path = os.path.join(config.PROCESSED_DATA_DIR, "transformer_model.ckpt")
    ms.save_checkpoint(model, ckpt_path)
    print(f"Transformer model saved to {ckpt_path}", flush=True)
    
    # 4. Evaluation
    print("Evaluating Transformer model on validation set...")
    model.set_train(False)
    
    num_val = len(seqs_val)
    val_preds = []
    val_targets_flat = []
    
    # Predict in batches
    for i in range(0, num_val, batch_size):
        s_b = ms.Tensor(seqs_val[i : i + batch_size], dtype=ms.int32)
        m_b = ms.Tensor(masks_val[i : i + batch_size], dtype=ms.int32)
        
        logits = model(s_b, m_b)
        probs = ops.sigmoid(logits).asnumpy()
        val_preds.append(probs)
        
    val_preds = np.concatenate(val_preds, axis=0)
    
    # Evaluation Metrics
    # Since we do multi-label classification, we can calculate ROC AUC and F1-score
    # For ROC AUC, we average AUC over validation users
    user_aucs = []
    user_f1s = []
    
    print("Computing validation metrics...")
    for j in range(num_val):
        actual_list = targets_val[j]
        actual_set = set(actual_list)
        
        if len(actual_set) == 0:
            continue
            
        probs_j = val_preds[j]
        
        # Create true binary label vector
        y_true_j = np.zeros(num_products_vocab, dtype=int)
        actual_list_valid = [p for p in actual_list if p < num_products_vocab]
        y_true_j[actual_list_valid] = 1
        
        try:
            auc = roc_auc_score(y_true_j, probs_j)
            user_aucs.append(auc)
        except ValueError:
            # Skip if user has no positive labels or only positive labels
            pass
            
        # Top 10 predictions F1-Score (standard metric for recommenders)
        # Select products with highest probabilities
        top_k = 10
        top_indices = np.argsort(probs_j)[-top_k:]
        pred_set = set(top_indices.tolist())
        
        intersection = len(actual_set.intersection(pred_set))
        if len(pred_set) == 0 or len(actual_set) == 0:
            f1 = 0.0
        else:
            precision = intersection / len(pred_set)
            recall = intersection / len(actual_set)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        user_f1s.append(f1)
        
    mean_auc = np.mean(user_aucs)
    mean_f1 = np.mean(user_f1s)
    print(f"Validation Mean User ROC AUC: {mean_auc:.4f}")
    print(f"Validation Mean User F1-score (Top 10): {mean_f1:.4f}")
    
    # Save validation metrics
    metrics = {
        'roc_auc': mean_auc,
        'f1_score': mean_f1
    }
    with open(os.path.join(config.PROCESSED_DATA_DIR, "transformer_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
        
    return model, metrics

if __name__ == "__main__":
    train_transformer()
