import os
import pickle
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import gradio as gr
from src import config
from src.matrix_factorization import load_mf_model_and_mappings
from src.transformer_model import SequentialTransformer
from src.xgboost_model import extract_features_and_candidates

# Global caches
print("Loading products mapping...")
products_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "products.csv"))
prod_id_to_name = dict(zip(products_df['product_id'], products_df['product_name']))
prod_name_to_id = dict(zip(products_df['product_name'], products_df['product_id']))

# Popular products (top 50 in raw prior)
prior_df = pd.read_csv(config.PRIOR_PRODUCTS_SAMPLE_PATH)
popular_ids = prior_df['product_id'].value_counts().head(50).index.tolist()
popular_products = [prod_id_to_name[pid] for pid in popular_ids if pid in prod_id_to_name]

# Load MF model and mappings
print("Loading Matrix Factorization model...")
mf_model, mf_mappings = load_mf_model_and_mappings()
num_products_vocab = mf_mappings['num_products_vocab']
user_to_idx = mf_mappings['user_to_idx']
idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

# Load Transformer model
print("Loading Sequential Transformer model...")
tf_model = SequentialTransformer(
    num_products=num_products_vocab,
    embedding_dim=config.TRANSFORMER_EMBEDDING_DIM,
    num_heads=config.TRANSFORMER_NUM_HEADS,
    num_layers=config.TRANSFORMER_NUM_LAYERS,
    max_seq_len=config.MAX_SEQ_LEN
)
ms.load_checkpoint(os.path.join(config.PROCESSED_DATA_DIR, "transformer_model.ckpt"), tf_model)
tf_model.set_train(False)

# Load XGBoost model
print("Loading XGBoost model and candidates...")
with open(os.path.join(config.PROCESSED_DATA_DIR, "xgb_model.pkl"), "rb") as f:
    xgb_model, feature_cols = pickle.load(f)

# Load all candidate features
candidates_all = extract_features_and_candidates()
# Get a list of valid user IDs (from validation set)
val_users = candidates_all[candidates_all['split'] == 'val']['user_id'].unique()
# Sort for presentation
sample_user_ids = sorted(list(val_users))[:20]

def make_html_cards(recommendations):
    """
    Generate beautiful HTML cards with product images from loremflickr.
    """
    html = '<div style="display: flex; flex-direction: column; gap: 12px; max-height: 600px; overflow-y: auto; padding: 4px;">'
    for name, score_text, val in recommendations:
        # Clean product name to make a relevant image search query
        query_words = []
        for w in name.split():
            w_clean = "".join(c for c in w if c.isalnum()).lower()
            # Filter out generic stop words in Instacart dataset
            if w_clean not in ['organic', 'natural', 'with', 'and', 'fresh', 'free', 'gluten', 'fat', 'original', 'classic', 'sweet', 'low']:
                query_words.append(w_clean)
        # Use first two descriptive keywords, default to "grocery"
        query = ",".join(query_words[:2]) if query_words else "grocery"
        img_url = f"https://loremflickr.com/120/120/{query},food/all"
        
        html += f"""
        <div style="display: flex; align-items: center; padding: 10px; border-radius: 10px; background: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.06); border: 1px solid #eaeaea; transition: transform 0.2s;">
            <img src="{img_url}" alt="{name}" style="width: 55px; height: 55px; object-fit: cover; border-radius: 8px; margin-right: 12px; background: #fafafa; border: 1px solid #f0f0f0;"/>
            <div style="flex-grow: 1;">
                <div style="font-weight: 600; font-size: 13.5px; color: #222; line-height: 1.3;">{name}</div>
                <div style="color: #0070f3; font-size: 11.5px; margin-top: 5px; font-weight: 500;">{score_text}: {val:.4f}</div>
            </div>
        </div>
        """
    html += '</div>'
    return html

def get_user_history_html(user_id):
    user_id = int(user_id)
    # Find user's historical purchases in prior dataset
    orders = pd.read_csv(config.ORDERS_SAMPLE_PATH)
    user_orders = orders[orders['user_id'] == user_id]['order_id'].tolist()
    
    user_prior = prior_df[prior_df['order_id'].isin(user_orders)]
    history_counts = user_prior['product_id'].value_counts()
    
    html = '<div style="display: flex; flex-direction: column; gap: 8px; max-height: 500px; overflow-y: auto; padding: 4px;">'
    count_limit = 0
    for pid, count in history_counts.items():
        if count_limit >= 15:
            break
        name = prod_id_to_name.get(pid, f"Product {pid}")
        query_words = []
        for w in name.split():
            w_clean = "".join(c for c in w if c.isalnum()).lower()
            if w_clean not in ['organic', 'natural', 'with', 'and', 'fresh', 'free', 'gluten', 'fat', 'original', 'classic', 'sweet', 'low']:
                query_words.append(w_clean)
        query = ",".join(query_words[:2]) if query_words else "grocery"
        img_url = f"https://loremflickr.com/100/100/{query},food/all"
        
        html += f"""
        <div style="display: flex; align-items: center; padding: 8px; border-radius: 8px; background: #fdfdfd; border: 1px solid #f0f0f0; margin-bottom: 2px;">
            <img src="{img_url}" alt="{name}" style="width: 42px; height: 42px; object-fit: cover; border-radius: 6px; margin-right: 10px; background: #f0f0f0; border: 1px solid #e5e5e5;"/>
            <div style="flex-grow: 1;">
                <div style="font-weight: 500; font-size: 12.5px; color: #333; line-height: 1.2;">{name}</div>
                <div style="color: #666; font-size: 11px; margin-top: 2px;">Bought {count} times</div>
            </div>
        </div>
        """
        count_limit += 1
        
    if count_limit == 0:
        html += '<div style="color: #999; text-align: center; padding: 20px;">No prior history found.</div>'
    html += '</div>'
    return html

def make_predictions(user_id, selected_names):
    user_id = int(user_id)
    
    # 1. Matrix Factorization (NCF) Recommendations (Static for User)
    mf_recs_data = []
    if user_id in user_to_idx:
        u_idx = user_to_idx[user_id]
        user_cands = candidates_all[candidates_all['user_id'] == user_id]
        if not user_cands.empty:
            p_ids = user_cands['product_id'].values.astype(np.int32)
            u_idxs = np.full(len(p_ids), u_idx, dtype=np.int32)
            
            # Use raw product IDs directly
            u_tensor = ms.Tensor(u_idxs)
            p_tensor = ms.Tensor(p_ids)
            logits = mf_model(u_tensor, p_tensor)
            probs = ops.sigmoid(logits).asnumpy()
            
            top_indices = np.argsort(probs)[::-1][:10]
            for idx in top_indices:
                pid = p_ids[idx]
                name = prod_id_to_name.get(pid, f"Product {pid}")
                mf_recs_data.append((name, "NCF Score", float(probs[idx])))
                
    # 2. XGBoost Recommendations (Static for User)
    xgb_recs_data = []
    user_cands = candidates_all[(candidates_all['user_id'] == user_id)]
    if not user_cands.empty:
        X = user_cands[feature_cols].values
        probs = xgb_model.predict_proba(X)[:, 1]
        top_indices = np.argsort(probs)[::-1][:10]
        for idx in top_indices:
            pid = user_cands['product_id'].iloc[idx]
            name = prod_id_to_name.get(pid, f"Product {pid}")
            xgb_recs_data.append((name, "XGB F1 Prob", float(probs[idx])))

    # 3. Sequential Transformer Recommendations (Dynamic based on shopping sequence!)
    selected_ids = [prod_name_to_id[name] for name in selected_names if name in prod_name_to_id]
    
    if not selected_ids:
        # Build empty sequence
        seq = np.zeros((config.MAX_SEQ_LEN, config.MAX_PRODUCTS_PER_ORDER), dtype=np.int32)
        mask = np.zeros(config.MAX_SEQ_LEN, dtype=np.int32)
    else:
        # User selected some items! Build a custom sequence of 1 order
        products_padded = selected_ids[:config.MAX_PRODUCTS_PER_ORDER] + [0] * max(0, config.MAX_PRODUCTS_PER_ORDER - len(selected_ids))
        seq = np.zeros((config.MAX_SEQ_LEN, config.MAX_PRODUCTS_PER_ORDER), dtype=np.int32)
        seq[-1] = products_padded
        mask = np.zeros(config.MAX_SEQ_LEN, dtype=np.int32)
        mask[-1] = 1
        
    s_b = ms.Tensor([seq], dtype=ms.int32)
    m_b = ms.Tensor([mask], dtype=ms.int32)
    logits = tf_model(s_b, m_b)
    probs = ops.sigmoid(logits).asnumpy()[0]
    
    top_indices = np.argsort(probs)[::-1][:10]
    tf_recs_data = []
    for idx in top_indices:
        name = prod_id_to_name.get(idx, f"Product {idx}")
        tf_recs_data.append((name, "Next Prob", float(probs[idx])))

    return (
        make_html_cards(tf_recs_data),
        make_html_cards(xgb_recs_data) if xgb_recs_data else '<div style="color: #999; text-align: center; padding: 20px;">No candidates.</div>',
        make_html_cards(mf_recs_data) if mf_recs_data else '<div style="color: #999; text-align: center; padding: 20px;">No NCF candidates.</div>'
    )

# Create Gradio interface with custom CSS to limit dropdown heights and ensure scrollability
with gr.Blocks(title="Instacart Market Basket Recommender Sandbox", css="""
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* Restrict dropdown menu height and add vertical scrollbars to prevent clipping */
    .dropdown-menu, .select-options, ul.options, div.options {
        max-height: 200px !important;
        overflow-y: auto !important;
    }
""") as demo:
    gr.Markdown("# Instacart Market Basket Recommender Sandbox")
    gr.Markdown(
        "Simulate shopping behavior and compare NCF (static user embedding), "
        "XGBoost (static engineered features), and the **Sequential Transformer (dynamic cart-sequence-based)**."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Select User to Simulate")
            user_id_input = gr.Dropdown(
                choices=[str(uid) for uid in sample_user_ids],
                label="Simulated User ID",
                value=str(sample_user_ids[0])
            )
            gr.Markdown("#### User Prior Purchases History")
            user_history_output = gr.HTML()
            
        with gr.Column(scale=2):
            gr.Markdown("### Step 2: Add Items to Current Cart (Build Shopping Sequence)")
            cart_items = gr.Dropdown(
                choices=list(prod_name_to_id.keys())[:500] + popular_products,
                multiselect=True,
                label="Add Items to Cart",
                info="Type to search and select products to add to your custom cart."
            )
            
            predict_btn = gr.Button("Predict Next Purchase", variant="primary")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Sequential Transformer (Dynamic)")
            tf_output = gr.HTML()
            
        with gr.Column():
            gr.Markdown("### XGBoost Classifier (Static)")
            xgb_output = gr.HTML()
            
        with gr.Column():
            gr.Markdown("### Matrix Factorization NCF (Static)")
            mf_output = gr.HTML()
            
    # Bind events
    user_id_input.change(fn=get_user_history_html, inputs=user_id_input, outputs=user_history_output)
    predict_btn.click(
        fn=make_predictions,
        inputs=[user_id_input, cart_items],
        outputs=[tf_output, xgb_output, mf_output]
    )
    
    # Initialize history on load
    demo.load(fn=get_user_history_html, inputs=user_id_input, outputs=user_history_output)

if __name__ == "__main__":
    demo.launch()
