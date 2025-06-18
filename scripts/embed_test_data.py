#!/usr/bin/env python
"""
Compute and save embeddings for all test passages (brute‐force).
"""

import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scripts.config import EMBEDDINGS_PATH, PARQUET_PATH, DEVICE, SAVE_DIR
from scripts.train import TwoTowerModel
from scripts.dataloader import tokenize_and_map, PAD_IDX, MAX_PASSAGE_LEN


def main() -> None:
    """
    Load trained model, embed every test passage, and save:
      - test_passage_embs.npy : array of shape [N, H]
      - test_passage_texts.pkl : list of N raw passage strings
    """
    # Load model and weights
    emb_matrix = np.load(EMBEDDINGS_PATH)
    model = TwoTowerModel(emb_matrix).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, "two_tower_final.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Path to test split parquet
    test_parquet = PARQUET_PATH.replace("train", "test")

    # Read raw passages from parquet
    df = pd.read_parquet(test_parquet, engine="fastparquet")
    all_passages: List[List[str]] = df["passages.passage_text"].tolist()
    raw_texts: List[str] = [p for sublist in all_passages for p in sublist]

    embeddings: List[np.ndarray] = []

    # Embed each passage one at a time
    for text in tqdm(raw_texts, desc="Embedding test passages"):
        idxs = tokenize_and_map(text)
        truncated = idxs[:MAX_PASSAGE_LEN]
        padded = truncated + [PAD_IDX] * max(0, MAX_PASSAGE_LEN - len(truncated))
        tensor = torch.tensor([padded], dtype=torch.long, device=DEVICE)
        lengths = torch.tensor([min(len(idxs), MAX_PASSAGE_LEN)], device=DEVICE)
        with torch.no_grad():
            emb = model.passage_encoder(tensor, lengths)
        embeddings.append(emb.squeeze(0).cpu().numpy())

    emb_array = np.stack(embeddings)

    # Save embeddings and associated texts
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "test_passage_embs.npy"), emb_array)
    with open(os.path.join(SAVE_DIR, "test_passage_texts.pkl"), "wb") as f:
        pickle.dump(raw_texts, f)


if __name__ == "__main__":
    main()
