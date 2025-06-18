#ben old data loader

import pyarrow as pa
import pandas as pd
import numpy as np
import pickle

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from nltk.tokenize import wordpunct_tokenize
from typing import Dict, Any

from scripts.config import (
    PARQUET_PATH,
    WORD_TO_IDX_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_QUERY_LEN,
    MAX_PASSAGE_LEN,
    NUM_PASSAGES,
)


# 2. Load vocab
with open(WORD_TO_IDX_PATH, "rb") as f:
    WORD_TO_IDX = pickle.load(f)
PAD_IDX = WORD_TO_IDX["<pad>"]
UNK_IDX = WORD_TO_IDX["<unk>"]


# 3. Tokeniser + mapper
def tokenize_and_map(text: str):
    tokens = wordpunct_tokenize(text.lower())
    return [WORD_TO_IDX.get(tok, UNK_IDX) for tok in tokens]

# 4. Dataset
class TwoTowerDataset(Dataset):
    def __init__(self, parquet_path):
        # you need either pyarrow or fastparquet installed
        self.df = pd.read_parquet(parquet_path, engine="fastparquet")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_idxs = tokenize_and_map(row["query"])
        p_texts = row["passages.passage_text"]  # list of 10 strings
        p_idxs = [tokenize_and_map(p) for p in p_texts]
        return {"query_indices": q_idxs, "passage_indices": p_idxs}

# 5. Collate
def collate_fn(batch):
    B = len(batch)
    Np = len(batch[0]["passage_indices"])  # num positive docs, should be 10

    # Queries
    q_lens = torch.tensor(
        [min(len(item["query_indices"]), MAX_QUERY_LEN) for item in batch],
        dtype=torch.long
    )
    q_batch = torch.tensor([
        item["query_indices"][:MAX_QUERY_LEN] +
        [PAD_IDX] * max(0, MAX_QUERY_LEN - len(item["query_indices"]))
        for item in batch
    ], dtype=torch.long)

    # Some queries have more or less than 10 docs/passages so pad up or slice down to 10
    fixed_passages = []
    for item in batch:
        ps = item["passage_indices"][:NUM_PASSAGES]
        if len(ps) < NUM_PASSAGES:
            ps += [[]] * (NUM_PASSAGES - len(ps))
        fixed_passages.append(ps)

    #Pad/truncate tokens in each passage
    padded = []
    lengths = []
    for ps in fixed_passages:
        row_tokens = []
        row_lens   = []
        for p in ps:
            # first build the padded token list
            tokens = p[:MAX_PASSAGE_LEN] + [PAD_IDX] * max(0, MAX_PASSAGE_LEN - len(p))
            row_tokens.append(tokens)

            # now compute length from the original p,
            # but ensure it's at least 1 so pack() won't crash
            true_len = min(len(p), MAX_PASSAGE_LEN)
            row_lens.append(max(1, true_len))
        padded.append(row_tokens)
        lengths.append(row_lens)

    # Build tensors *once*
    p_batch = torch.tensor(padded,  dtype=torch.long)  # [B, 10, MAX_PASSAGE_LEN]
    p_lens  = torch.tensor(lengths, dtype=torch.long)  # [B, 10]

    return {
      "query":           q_batch,
      "query_lengths":   q_lens,
      "passages":        p_batch,
      "passage_lengths": p_lens,
    }

# 6. DataLoader
dataset  = TwoTowerDataset(PARQUET_PATH)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn
)

# 7. Smoke test
# batch = next(iter(dataloader))
# print("query:",            batch["query"].shape)          # (BATCH_SIZE, 32)
# print("query_lengths:",    batch["query_lengths"].shape)  # (BATCH_SIZE,)
# print("passages:",         batch["passages"].shape)       # (BATCH_SIZE,10,256)
# print("passage_lengths:",  batch["passage_lengths"].shape)# (BATCH_SIZE,10)
