import pyarrow as pa
import pandas as pd
import numpy as np
import pickle

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.tokenize import wordpunct_tokenize
from typing import Dict, Any

import wandb
from tqdm import tqdm

from scripts.dataloader import TwoTowerDataset, dataloader, collate_fn
from scripts.config import (
    EMBEDDINGS_PATH,
    PAD_IDX,
    UNK_IDX,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_QUERY_LEN,
    MAX_PASSAGE_LEN,
    NUM_PASSAGES,
    EMBED_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    FREEZE_EMBEDDINGS,
    LR,
    MARGIN,
    EPOCHS,
    DEVICE,
    WANDB_PROJECT,
    SAVE_DIR,
)

# Check torch can find gpu
print("CUDA available?:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


# ─── 2. RNN Encoder ────────────────────────────────────────────────────────────
class RNNEncoder(nn.Module):
    """Encodes a batch of token-ID sequences via an LSTM into fixed vectors."""
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        freeze_embeddings: bool = True
    ):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix).float(),
            freeze=freeze_embeddings,
            padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] token IDs
        lengths: [B] real lengths
        → returns [B, hidden_size] final hidden state
        """
        emb = self.embedding(x)  # [B, L, D]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        return h_n[-1]  # [B, hidden_size]


# ─── 3. Two-Tower Model ────────────────────────────────────────────────────────
class TwoTowerModel(nn.Module):
    """Two separate RNN encoders for queries and passages."""
    def __init__(self, embedding_matrix: np.ndarray):
        super().__init__()
        # freeze_embeddings=True by default; change later if you want to unfreeze
        self.query_encoder   = RNNEncoder(embedding_matrix,
                                            HIDDEN_SIZE,
                                            NUM_LAYERS,
                                            DROPOUT,
                                            freeze_embeddings=FREEZE_EMBEDDINGS)
        self.passage_encoder = RNNEncoder(embedding_matrix,
                                            HIDDEN_SIZE,
                                            NUM_LAYERS,
                                            DROPOUT,
                                            freeze_embeddings=FREEZE_EMBEDDINGS)

    def forward(
        self,
        queries: torch.Tensor,
        q_lens: torch.Tensor,
        passages: torch.Tensor,
        p_lens: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        queries: [B, MAX_Q_LEN]
        q_lens:  [B]
        passages: [B, Np, MAX_P_LEN]
        p_lens: [B, Np]
        → returns:
           q_emb: [B, H]
           p_emb: [B, Np, H]
        """
        B, Np, Lp = passages.shape
        # Encode queries
        q_emb = self.query_encoder(queries, q_lens)             # [B, H]
        # Encode passages (flatten then reshape)
        p_flat = passages.view(B * Np, Lp)
        p_len_flat = p_lens.view(B * Np)
        p_emb_flat = self.passage_encoder(p_flat, p_len_flat)   # [B*Np, H]
        p_emb = p_emb_flat.view(B, Np, -1)                      # [B, Np, H]
        return q_emb, p_emb


# ─── 5. Metrics ────────────────────────────────────────────────────────────────
def full_batch_triplet_loss(
    q_emb: torch.Tensor,          # [B, H]
    p_emb: torch.Tensor,          # [B, 10, H]
    loss_fn: nn.TripletMarginLoss
) -> torch.Tensor:
    """
    For MS MARCO: treat all 10 p_emb[:,i] as positives,
    and for each one sample one negative from another query in the batch.
    Returns scalar loss over B*10 triplets.
    """
    B, Np, H = p_emb.shape      # Np==10

    # 1) Flatten positives & repeat queries
    q_flat   = q_emb.unsqueeze(1).expand(B, Np, H).reshape(-1, H)  # [B*10, H]
    pos_flat = p_emb.reshape(-1, H)                                # [B*10, H]

    # 2) Build index mask so we only sample from OTHER queries
    total = B * Np
    idxs  = torch.arange(total, device=p_emb.device)
    q_idxs = idxs // Np                     # integer query‐ids for each pos
    mask   = q_idxs.unsqueeze(1) != q_idxs.unsqueeze(0)  # [total, total]

    # 3) For each of the B*10 positives pick one random negative
    all_cands = pos_flat                   # full list of B*10 “positives” as candidates
    neg_list = []
    for i in range(total):
        valid      = idxs[mask[i]]         # all candidates not from the same query
        choice     = valid[torch.randint(len(valid), (1,))]  # pick one
        neg_list.append(all_cands[choice])
    neg_flat = torch.cat(neg_list, dim=0)  # [B*10, H]

    # 4) TripletMarginLoss on all B*10 triplets at once
    return loss_fn(q_flat, pos_flat, neg_flat)



def recall_at_k_inbatch(q_emb: torch.Tensor,
                        p_emb: torch.Tensor,
                        k: int = 1,
                        num_negs: int = None) -> float:
    """
    In-batch recall@k when each query has multiple positives.
    q_emb: [B, H]
    p_emb: [B, Np, H]  # Np = 10 positives
    num_negs: how many negatives to sample per query (from other queries' positives).
      If None, sample up to B*10 - Np.
    Returns average recall@k over batch.
    """
    B, Np, H = p_emb.shape
    sims_list = []
    hits = []
    # Flatten all candidates for easy indexing
    all_pos_flat = p_emb.reshape(-1, H)  # [B*10, H]
    total = B * Np
    idxs = torch.arange(total, device=p_emb.device)
    q_idxs = idxs // Np

    for i in range(B):
        # 1) get query vector
        qv = q_emb[i]  # [H]
        # 2) positives for this query
        pos_embs = p_emb[i]  # [Np, H]
        # 3) sample negatives from other queries
        # indices of all other positives: those j where j//Np != i
        mask = (q_idxs != i)
        candidates = all_pos_flat[mask]  # [(B*10 - Np), H]
        if num_negs is None:
            neg_embs = candidates  # use all
        else:
            # random sample num_negs
            perm = torch.randperm(candidates.size(0), device=p_emb.device)
            neg_embs = candidates[perm[:num_negs]]  # [num_negs, H]

        # 4) build combined set: first all positives, then sampled negs
        combined = torch.cat([pos_embs, neg_embs], dim=0)  # [Np+num_negs, H]
        # 5) compute sims
        sims = combined.matmul(qv)  # [Np+num_negs]
        # 6) check if any of the positives (indices 0..Np-1) are in top-k
        topk = sims.topk(k).indices  # indices in 0..Np+num_negs-1
        hit = (topk < Np).any().float()  # True if at least one positive in top-k
        hits.append(hit)
    return torch.stack(hits).mean().item()


def gap_inbatch(q_emb: torch.Tensor,
                p_emb: torch.Tensor,
                num_negs: int = None) -> float:
    """
    Average gap over all positives in batch.
    For each (i, j) positive: gap = sim(q_i, p_emb[i,j]) - max(sim(q_i, negs_from_other_queries)).
    """
    B, Np, H = p_emb.shape
    all_pos_flat = p_emb.reshape(-1, H)
    total = B * Np
    idxs = torch.arange(total, device=p_emb.device)
    q_idxs = idxs // Np
    gaps = []
    for idx in range(total):
        qi = idx // Np
        # query vector
        qv = q_emb[qi]  # [H]
        # this positive
        pv = all_pos_flat[idx]  # [H]
        # sample negatives from other queries
        mask = (q_idxs != qi)
        candidates = all_pos_flat[mask]  # [(B*10 - 10), H]
        if num_negs is not None:
            perm = torch.randperm(candidates.size(0), device=p_emb.device)
            candidates = candidates[perm[:num_negs]]
        # sims
        pos_sim = torch.dot(qv, pv)
        neg_sims, _ = (candidates.matmul(qv)).max(dim=0)  # scalar
        gaps.append(pos_sim - neg_sims)
    return torch.stack(gaps).mean().item()


# ─── 7. Putting it all together ───────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load pretrained embeddings
    emb_matrix = np.load(EMBEDDINGS_PATH)

    # 2) Build model, optimizer, loss
    model     = TwoTowerModel(emb_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.TripletMarginLoss(margin=MARGIN)

    # 3) Init wandb
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "margin": MARGIN,
            "hidden_size": HIDDEN_SIZE,
        }
    )

    # 4) Main training loop (per-batch logging + multi-neg loss)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # move to device
            q   = batch["query"].to(DEVICE)
            ql  = batch["query_lengths"].to(DEVICE)
            p   = batch["passages"].to(DEVICE)
            pl  = batch["passage_lengths"].to(DEVICE)

            # encode + normalise
            q_emb, p_emb = model(q, ql, p, pl)
            # compute *all-nine* negative loss
            loss = full_batch_triplet_loss(q_emb, p_emb, loss_fn)

            # backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # For metrics, normalise to be cosine‐style:
            qn = F.normalize(q_emb, dim=1)
            pn = F.normalize(p_emb, dim=2)

            # 3) Compute in-batch recall@k and gap
            rec1 = recall_at_k_inbatch(qn, pn, k=1, num_negs=50)  # e.g. sample 50 negatives
            rec5 = recall_at_k_inbatch(qn, pn, k=5, num_negs=50)
            gap  = gap_inbatch(qn, pn, num_negs=50)


            # log per batch
            wandb.log({
                "epoch":      epoch,
                "batch_idx":  batch_idx,
                "batch_loss": loss.item(),
                "recall@1":   rec1,
                "recall@5":   rec5,
                "gap":        gap,
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} done — avg loss {avg_loss:.4f}")

    # 5) Save final checkpoint
    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, "two_tower_final.pt")
    )


