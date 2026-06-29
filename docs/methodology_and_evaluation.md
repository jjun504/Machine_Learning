# Project Report Notes: Methodology, Features, and Evaluation

This document compiles the core theoretical explanations, model architectures, feature engineering details, and metric evaluations discussed during our design session. You can copy, adapt, and expand these sections directly into your final 35-page project report: **“Benchmarking and Deployment of Different Machine Learning Strategies in Smart Retail Recommendations”**.

---

## 1. Problem Formulation: Next-Basket Recommendation (NBR)

In smart retail, recommendation is formulated not as predicting a single next click, but as **Next-Basket Recommendation (NBR)**—predicting the exact set of items a user will purchase in their next checkout.

### 1.1 Chronological Split: Prior vs. Train
To simulate real-world prediction without cheating, the dataset enforces a strict chronological boundary:
*   **Prior Dataset (`prior`)**: All historical orders of a user from order $1$ to $T-1$. This data is used for **feature extraction** and profiling user habits (e.g., how often they buy milk).
*   **Train Dataset (`train`)**: The user's latest order at time $T$. This acts as the **ground truth** (the standard answer).
*   **Data Leakage Prevention**: Features must *only* be calculated using the `prior` set. If target order $T$ is mixed into feature calculation, the model would cheat (leakage) by knowing what was bought in order $T$ before predicting it.

### 1.2 User-Level Train/Validation Split
We sample 5,000 users who have target orders in the `train` set and split them into **80% training users (4,000)** and **20% validation users (1,000)**:
*   **User-Level Partitioning**: A user's entire history and target order are kept together in either the training split or the validation split. 
*   **Zero-Exposure Validation**: The 1,000 validation users are completely hidden during model training. This ensures the benchmark evaluates the model's ability to generalize to unseen customers.

---

## 2. Model Selection and Architectures

To satisfy the guidelines of benchmarking different machine learning strategies, we implement three distinct paradigms:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ML Strategies Comparison                           │
├─────────────────────────┬────────────────────────┬──────────────────────┤
│ Strategy A (NCF)        │ Strategy B (XGBoost)   │ Strategy C (TF)      │
├─────────────────────────┼────────────────────────┼──────────────────────┤
│ Collaborative Filtering │ Tabular Classifier     │ Sequence Modeling    │
│ Embeddings Co-occurrence│ Hand-crafted Features  │ Self-Attention       │
│ User/Item latent space  │ Decision Tree Boosting │ Chronological Baskets│
└─────────────────────────┴────────────────────────┴──────────────────────┘
```

### 2.1 Strategy A: Neural Collaborative Filtering (NCF) in MindSpore
*   **Concept**: Maps users and products to a shared 64-dimensional latent space (embeddings).
*   **Architecture**: Subclasses `mindspore.nn.Cell`. Predicts affinity by taking the dot product of the user and product embeddings, adjusted by user and product biases:
    $$\hat{y}_{u,p} = \text{Embedding}_u \cdot \text{Embedding}_p + b_u + b_p$$
*   **Negative Sampling**: Since transaction history only contains positive purchases (implicit feedback, $y=1$), we perform negative sampling (4 negatives per positive) during training to teach the model what items the user did not buy.

### 2.2 Strategy B: Extreme Gradient Boosting (XGBoost)
*   **Concept**: An ensemble of Classification and Regression Trees (CART) trained sequentially using Gradient Boosting.
*   **Model Stacking**: We feed the predictions of the trained NCF model (`mf_score`) into the XGBoost feature set, combining global collaborative patterns with local temporal features.
*   **Base Learner**: Composed of 150 regression trees (`n_estimators=150`) of depth 6. Each tree fits the residual errors (mistakes) of the preceding trees, optimizing the binary logloss.

### 2.3 Strategy C: Sequential Transformer Encoder in MindSpore
*   **Concept**: Captures the chronological rhythm of purchase baskets.
*   **Architecture**: Aggregates product embeddings to represent each historical basket, adds learnable positional encodings, and feeds the sequence of 10 baskets through a Multi-Head Self-Attention `nn.TransformerEncoder` block to extract the user's hidden state.
*   **Output**: Projects the final hidden state to the entire product vocabulary (49k classes) using a dense projection layer.

---

## 3. Feature Engineering (XGBoost)

XGBoost's superior performance is driven by 12 hand-crafted features extracted from the `prior` dataset, divided into three layers:

1.  **User Features (User Habits)**:
    *   `user_total_orders`: User's total order count (indicates loyalty/regularity).
    *   `user_avg_basket_size`: Average items per order.
    *   `user_reorder_rate`: Proportion of reordered products.
    *   `user_avg_days_since_prior_order`: Average frequency of shopping trips.
2.  **Product Features (Product Popularity)**:
    *   `product_total_orders`: Total platform sales of the product.
    *   `product_reorder_rate`: platform-wide repeat purchase probability (staples vs. one-off items).
    *   `product_avg_add_to_cart`: Average order in which the item is added to the cart.
3.  **User-Product Interaction Features (Personalized Habits)**:
    *   `up_total_orders`: Total times the user bought this product.
    *   `up_purchase_rate`: Proportion of the user's orders containing this product.
    *   `up_last_order_distance`: **Crucial recency feature** (Current user orders minus last order number containing the product). A distance of 0 means bought in the last order.
    *   `up_avg_add_to_cart`: Average position the user puts this product in their cart.
4.  **Collaborative Feature**:
    *   `mf_score`: The static affinity probability outputted by the MindSpore NCF model.

---

## 4. Integrated Learning Concepts: Bagging vs. Boosting

In the methodology section, justify selecting XGBoost (Boosting) over Random Forest (Bagging) using these integration details:

*   **Bagging (Random Forest - Parallel Voting)**:
    *   *Mechanism*: Trains multiple deep decision trees independently in parallel on random bootstrap samples. Predictions are aggregated by majority vote.
    *   *Purpose*: Focuses on reducing **variance** (overfitting) by averaging out extreme errors.
*   **Boosting (XGBoost - Sequential Correction)**:
    *   *Mechanism*: Trains shallow regression trees sequentially. Each tree is trained to predict the residual errors (gradients) of the ensemble up to that point.
    *   *Purpose*: Focuses on reducing **bias** (underfitting) by constantly targeting hard-to-classify samples.
*   **Selection Rationale**: In next-basket recommendation, the dataset is highly imbalanced and driven by sparse, dominant features (like recency). Boosting is far more effective than Bagging at focusing on these sparse signals through sequential error correction.

### 4.1 Hyperparameter Tuning (Grid Search)
To satisfy the project requirements of optimization, we perform a Grid Search over key hyperparameters of the XGBoost classifier:
*   `max_depth` (Tree depth limit): `[4, 6, 8]`
*   `learning_rate` (Gradient step shrinkage): `[0.05, 0.1, 0.2]`

#### Grid Search Results Table

| max_depth | learning_rate | ROC AUC | Best F1-Score | Best Threshold | Training Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 0.05 | **0.8218** | **0.3693** | 0.20 | 3.0s |
| 6 | 0.10 | 0.8206 | 0.3691 | 0.20 | 2.0s |
| 4 | 0.10 | 0.8219 | 0.3681 | 0.20 | 1.4s |
| 6 | 0.05 | 0.8223 | 0.3679 | 0.20 | 2.5s |
| 4 | 0.20 | 0.8202 | 0.3670 | 0.20 | 1.7s |
| 8 | 0.10 | 0.8177 | 0.3640 | 0.20 | 2.6s |
| 8 | 0.05 | 0.8211 | 0.3639 | 0.20 | 2.8s |
| 6 | 0.20 | 0.8165 | 0.3614 | 0.20 | 2.1s |
| 8 | 0.20 | 0.8096 | 0.3558 | 0.20 | 2.5s |

#### Quantitative Insights (Overfitting Observation)
*   **The Optimal Configuration**: `max_depth = 4` and `learning_rate = 0.05` achieves the best validation F1-score of **`0.3693`** and ROC AUC of **`0.8218`**.
*   **The Overfitting Penalty**: As we increase the tree complexity (`max_depth = 8`) and speed up learning (`learning_rate = 0.20`), the model performance drops significantly (F1 falls to `0.3558` and AUC falls to `0.8096`). This is a classic case of **model overfitting** on the training features, leading to poorer generalization on the unseen validation users.

---

## 5. Benchmarking and Evaluation Metrics

To evaluate recommendation effectiveness, we implement 7 offline metrics on the validation set of 1,000 users (recommending $K=10$ items):

1.  **ROC AUC (Ranking Quality)**: Probability that the model ranks a randomly chosen positive item higher than a negative one.
2.  **F1@10 (Balanced Quality)**: The harmonic mean of Precision@10 and Recall@10, representing the overlap quality between the recommended basket and the actual basket.
3.  **Precision@10 (Accuracy)**: Percentage of recommended items that were actually purchased ($\text{Hits} / 10$).
4.  **Recall@10 (Coverage)**: Percentage of actual purchases captured by the recommended list ($\text{Hits} / \text{Actual size}$).
5.  **Hit Rate@10 (Utility)**: Proportion of users who received at least one correct recommendation.
6.  **NDCG@10 (Position-Aware Gain)**: Normalized Discounted Cumulative Gain, penalizing the model if correct items are ranked lower in the top-10 list.
7.  **MRR (Mean Reciprocal Rank)**: Reciprocal of the rank of the first correct recommendation ($\text{Mean of } 1/\text{rank}$).

### 5.1 Benchmarking Results Table (Apples-to-Apples Candidate Evaluation)

| Model | ROC AUC | F1@10 | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 | MRR | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matrix Factorization (NCF)** | `0.5494` | `0.2173` | `0.1568` | `0.3535` | `0.7661` | `0.2775` | `0.3782` | ~1.13 min |
| **XGBoost (Classifier)** | **`0.8222`** | **`0.3911`** | **`0.2963`** | **`0.5750`** | **`0.9228`** | **`0.5338`** | **`0.6702`** | **~0.09 min** |
| **Sequential Transformer** | `0.5303` | `0.1987` | `0.1420` | `0.3308` | `0.7376` | `0.2676` | `0.4060` | ~1.19 min |

---

## 6. Academic Discussion & Quantitative Insights

### 6.1 Why XGBoost Outperforms Deep Learning in Groceries
1.  **Temporal/Habitual Dominance**: Grocery retail is heavily cyclical. Explicit time features (like `up_last_order_distance` and `up_purchase_rate`) capture this cycle immediately. XGBoost splits on these features directly. 
2.  **Movie vs. Grocery Recommendation**: In movies (e.g., Netflix), users rarely watch the same movie twice (reorder = 0), so static collaborative filtering (MF) is king. In groceries, users reorder the same staples weekly (reorder = 1), making temporal tabular models far more predictive.

### 6.2 The ROC AUC Mathematical Nuance: Candidate vs. Vocabulary Space
In your report, explain the difference in validation search spaces:
*   **Candidate-restricted (Fair) Space**: Compares candidates (products the user has bought before). These are **hard negatives** because they match the user's general taste. Distinguishing what they bought *today* vs. *last week* is difficult, yielding a lower ROC AUC (`0.5303` for Transformer).
*   **Vocabulary-wide Space**: Compares positive items against all 49k products (including baby diaper items for single bachelors). These are **easy negatives** which get near-zero scores. Because 99% of the pool is composed of these easy negatives, the ROC AUC naturally inflates (`0.5572` for Transformer), even though the F1-score drops due to massive competitor noise.

### 6.3 Sequential Transformer MRR Advantage
While the Transformer's overall F1-score (`0.1987`) is slightly lower than NCF (`0.2173`), it achieves a **significantly higher Mean Reciprocal Rank (MRR = 0.4060 vs. 0.3782)**. This proves that the multi-head self-attention layer successfully captures sequence transitions, placing the *first correct item* significantly higher in the recommended list than static collaborative filtering.
