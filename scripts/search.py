#!/usr/bin/env python
"""
Search script for Two-Tower Retrieval.

Loads precomputed test-passage embeddings and the trained model,
then encodes a user query, performs brute-force similarity search,
and prints the top-k passages.
"""

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch

from scripts.config import (
    DEVICE,
    SAVE_DIR,
    PAD_IDX,
    MAX_QUERY_LEN,
    EMBEDDINGS_PATH
)
from scripts.train import TwoTowerModel
from scripts.dataloader import tokenize_and_map


def load_passage_data() -> Tuple[np.ndarray, List[str]]:
    """
    Load precomputed passage embeddings and raw texts.

    Returns:
        emb_array: [N, H] numpy array of passage embeddings
        texts: list of N passage strings
    """
    emb_path = os.path.join(SAVE_DIR, "test_passage_embs.npy")
    txt_path = os.path.join(SAVE_DIR, "test_passage_texts.pkl")
    emb_array = np.load(emb_path)
    with open(txt_path, "rb") as f:
        texts = pickle.load(f)
    return emb_array, texts


def embed_query(
    model: TwoTowerModel,
    query: str
) -> np.ndarray:
    """
    Tokenize and embed a single query string.

    Args:
        model: TwoTowerModel with loaded weights
        query: raw query string

    Returns:
        numpy array of shape [H]
    """
    # tokenize and map to indices, add padding/truncation
    idxs = tokenize_and_map(query)
    truncated = idxs[:MAX_QUERY_LEN]
    padded = truncated + [PAD_IDX] * max(0, MAX_QUERY_LEN - len(truncated))
    tensor = torch.tensor([padded], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([min(len(idxs), MAX_QUERY_LEN)], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        q_emb = model.query_encoder(tensor, lengths)
    return q_emb.squeeze(0).cpu().numpy()


def search(
    q_vec: np.ndarray,
    p_embs: np.ndarray,
    texts: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Compute dot-product similarities and return top-k passages.

    Args:
        q_vec: [H] query embedding
        p_embs: [N, H] passage embeddings
        texts: list of N passage strings
        top_k: number of results

    Returns:
        List of (passage, score) sorted by descending score
    """
    # dot-product
    sims = p_embs.dot(q_vec)  # [N]
    # top-k indices
    idxs = np.argpartition(-sims, top_k - 1)[:top_k]
    # sort them
    top_idxs = idxs[np.argsort(-sims[idxs])]
    return [(texts[i], float(sims[i])) for i in top_idxs]


def main() -> None:
    """Parse arguments, load data and model, run search, and print results."""
    parser = argparse.ArgumentParser(description="Search Two-Tower Retrieval")
    parser.add_argument("--query", "-q", required=True, help="Query string to search")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of top results")
    args = parser.parse_args()

    # load passage database
    p_embs, texts = load_passage_data()

    # load model
    print("Loading model…")
    emb_matrix = np.load(EMBEDDINGS_PATH)   # your static GloVe+… file
    model = TwoTowerModel(emb_matrix).to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, "two_tower_ckpt1.pt")  # your actual checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # embed query and search
    q_vec = embed_query(model, args.query)
    print('Running search...')
    results = search(q_vec, p_embs, texts, top_k=args.top_k)

    # print
    print(f"\nTop {args.top_k} results for query: \"{args.query}\"\n")
    for rank, (passage, score) in enumerate(results, start=1):
        print(f"{rank}. (score={score:.4f}) {passage}\n")


if __name__ == "__main__":
    main()

