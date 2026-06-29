import os

# Root directory of the project
ROOT_DIR = r"c:\Users\chenj\OneDrive\Desktop\UUU\Machine Learning\Project\code"

# Raw and processed data paths
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Create processed directory if it is not exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Dataset configuration
# NUM_USERS = 1000  # too small for train
# NUM_USERS = 20000  # memory error on my computer
NUM_USERS = 5000  # Number of user to sample
RANDOM_SEED = 42

# Preprocessed CSV file paths
ORDERS_SAMPLE_PATH = os.path.join(PROCESSED_DATA_DIR, "orders_sample.csv")
PRIOR_PRODUCTS_SAMPLE_PATH = os.path.join(PROCESSED_DATA_DIR, "order_products__prior_sample.csv")
TRAIN_PRODUCTS_SAMPLE_PATH = os.path.join(PROCESSED_DATA_DIR, "order_products__train_sample.csv")
PRODUCTS_METADATA_PATH = os.path.join(RAW_DATA_DIR, "products.csv")

# Model Training Parameters
# 1. Matrix Factorization (MindSpore)
# MF_EMBEDDING_DIM = 32  # Tested 32, but 64 gets +0.015 AUC
# MF_EMBEDDING_DIM = 128  # 128 is too slow to train
MF_EMBEDDING_DIM = 64
MF_BATCH_SIZE = 2048
MF_EPOCHS = 5
# MF_LR = 0.01  # Fluctuates too much
# MF_LR = 0.001  # Converges too slow
MF_LR = 0.005

# 2. XGBoost
XGB_N_ESTIMATORS = 150
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# 3. Transformer (MindSpore)
# TRANSFORMER_EMBEDDING_DIM = 32  # Underfits
TRANSFORMER_EMBEDDING_DIM = 64
# TRANSFORMER_NUM_HEADS = 2
TRANSFORMER_NUM_HEADS = 4
# TRANSFORMER_NUM_LAYERS = 1  # F1 score is too low
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_BATCH_SIZE = 512
TRANSFORMER_EPOCHS = 5
TRANSFORMER_LR = 0.001
MAX_SEQ_LEN = 10  # Predict next order based on last 10 orders
MAX_PRODUCTS_PER_ORDER = 30  # Truncate orders to last 30 products
