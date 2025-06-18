"""
Configuration module for the Two-Tower Retrieval project.
Defines dataset paths, vocab indices, hyperparameters, and device settings.
"""

import os
import pickle
import torch

# ─── 1. Directory paths ───────────────────────────────────────────────────────
FILE_DIR: str = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR: str = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
DATA_DIR: str = os.path.join(ROOT_DIR, "data")
SCRIPTS_DIR: str = os.path.join(ROOT_DIR, "scripts")

# ─── 2. Data file paths ──────────────────────────────────────────────────────
PARQUET_PATH: str     = os.path.join(DATA_DIR, "ms_marco_train.parquet")
WORD_TO_IDX_PATH: str = os.path.join(DATA_DIR, "word_to_idx.pkl")
EMBEDDINGS_PATH: str  = os.path.join(DATA_DIR, "embeddings.npy")

# ─── 3. Vocab & pad/unk ───────────────────────────────────────────────────────
WORD_TO_IDX: dict     = pickle.load(open(WORD_TO_IDX_PATH, "rb"))
PAD_IDX: int          = WORD_TO_IDX["<pad>"]
UNK_IDX: int          = WORD_TO_IDX["<unk>"]

# ─── 4. DataLoader parameters ────────────────────────────────────────────────
BATCH_SIZE: int       = 256  # samples per batch
NUM_WORKERS: int      = 4   # DataLoader processes
MAX_QUERY_LEN: int    = 32  # max tokens in a query
MAX_PASSAGE_LEN: int  = 256 # max tokens in a passage
NUM_PASSAGES: int     = 10  # expected number of passages per query

# ─── 5. Model hyperparameters ─────────────────────────────────────────────────
EMBED_DIM: int            = 200
HIDDEN_SIZE: int          = 256
NUM_LAYERS: int           = 1
DROPOUT: float            = 0.1
FREEZE_EMBEDDINGS: bool   = False

# ─── 6. Training hyperparameters ─────────────────────────────────────────────
LR: float     = 1e-3
MARGIN: float = 1.0
EPOCHS: int   = 1

# ─── 7. Device configuration ─────────────────────────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 8. Weights & Biases settings ─────────────────────────────────────────────
WANDB_PROJECT: str = "two-tower-retrieval"

# ─── 9. Save dir ─────────────────────────────────────────────

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)
