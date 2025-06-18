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

from scripts.dataloader import train_loader, val_loader, test_loader
from torch.utils.data import DataLoader # for type hinting

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

# eval function for jsut a single batch
def eval_one_batch(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    loss_fn: nn.TripletMarginLoss
) -> Dict[str, float]:
    model.eval()
    q, ql = batch["query"].to(DEVICE), batch["query_lengths"].to(DEVICE)
    p, pl = batch["passages"].to(DEVICE), batch["passage_lengths"].to(DEVICE)
    with torch.no_grad():
        q_emb, p_emb = model(q, ql, p, pl)
        loss = full_batch_triplet_loss(q_emb, p_emb, loss_fn).item()
        qn, pn = F.normalize(q_emb, dim=1), F.normalize(p_emb, dim=2)
        r1 = recall_at_k_inbatch(qn, pn, k=1, num_negs=50)
        r5 = recall_at_k_inbatch(qn, pn, k=5, num_negs=50)
        g  = gap_inbatch(qn, pn, num_negs=50)
    model.train()
    return {"loss": loss, "recall@1": r1, "recall@5": r5, "gap": g}

# Eval function o hwole val adn test set
def evaluate(loader: DataLoader,
             model: nn.Module,
             loss_fn: nn.TripletMarginLoss) -> Dict[str, float]:
  """
  Run one pass over loader and return average metrics.
  """
  model.eval()
  losses, r1s, r5s, gaps = [], [], [], []
  with torch.no_grad():
    for batch in loader:
      q, ql = batch["query"].to(DEVICE), batch["query_lengths"].to(DEVICE)
      p, pl = batch["passages"].to(DEVICE), batch["passage_lengths"].to(DEVICE)
      q_emb, p_emb = model(q, ql, p, pl)
      losses.append(full_batch_triplet_loss(q_emb, p_emb, loss_fn).item())
      qn, pn = F.normalize(q_emb, dim=1), F.normalize(p_emb, dim=2)
      r1s.append(recall_at_k_inbatch(qn, pn, k=1, num_negs=50))
      r5s.append(recall_at_k_inbatch(qn, pn, k=5, num_negs=50))
      gaps.append(gap_inbatch(qn, pn, num_negs=50))
  model.train()
  return {
    "loss":    sum(losses) / len(losses),
    "recall@1": sum(r1s)    / len(r1s),
    "recall@5": sum(r5s)    / len(r5s),
    "gap":      sum(gaps)   / len(gaps),
  }

def main() -> None:
    """Train two-tower with periodic batch-eval and end-of-epoch full eval."""
    # 1) Load embeddings, build model/optimizer/loss
    emb_matrix = np.load(EMBEDDINGS_PATH)
    model = TwoTowerModel(emb_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN)

    # 2) Prepare single-batch iterators for on-the-fly eval
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    # 3) Init WandB
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "margin": MARGIN,
            "hidden_size": HIDDEN_SIZE,
        },
    )

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            step += 1
            # move to device
            q, ql = batch["query"].to(DEVICE), batch["query_lengths"].to(DEVICE)
            p, pl = batch["passages"].to(DEVICE), batch["passage_lengths"].to(DEVICE)

            # forward + loss
            q_emb, p_emb = model(q, ql, p, pl)
            loss = full_batch_triplet_loss(q_emb, p_emb, loss_fn)

            # backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # compute train metrics
            qn, pn = F.normalize(q_emb, dim=1), F.normalize(p_emb, dim=2)
            rec1 = recall_at_k_inbatch(qn, pn, k=1, num_negs=50)
            rec5 = recall_at_k_inbatch(qn, pn, k=5, num_negs=50)
            gap = gap_inbatch(qn, pn, num_negs=50)

            # assemble train logs
            logs: Dict[str, float] = {
                "loss/train": loss.item(),
                "recall@1/train": rec1,
                "recall@5/train": rec5,
                "gap/train": gap,
            }

            # every 50 steps, grab one batch from val & test and eval
            if step % 50 == 0:
                try:
                    vb = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    vb = next(val_iter)

                try:
                    tb = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    tb = next(test_iter)

                val_m = eval_one_batch(vb, model, loss_fn)
                test_m = eval_one_batch(tb, model, loss_fn)

                logs.update({
                    "loss/val": val_m["loss"],
                    "recall@1/val": val_m["recall@1"],
                    "recall@5/val": val_m["recall@5"],
                    "gap/val": val_m["gap"],
                    "loss/test": test_m["loss"],
                    "recall@1/test": test_m["recall@1"],
                    "recall@5/test": test_m["recall@5"],
                    "gap/test": test_m["gap"],
                })

            # log everything under same step
            wandb.log(logs, step=step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} done — avg loss {avg_loss:.4f}")

        # Save checkpoint immediately
        ckpt_path = os.path.join(SAVE_DIR, f"ckpt_epoch_{epoch}.pt")
        print(f"Saving checkpoint: {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

        # end-of-epoch full eval
        print("Running end-of-epoch evaluation")
        full_val = evaluate(val_loader, model, loss_fn)
        full_test = evaluate(test_loader, model, loss_fn)
        wandb.log({
            **{f"loss/val": full_val["loss"],
               f"recall@1/val": full_val["recall@1"],
               f"recall@5/val": full_val["recall@5"],
               f"gap/val": full_val["gap"]},
            **{f"loss/test": full_test["loss"],
               f"recall@1/test": full_test["recall@1"],
               f"recall@5/test": full_test["recall@5"],
               f"gap/test": full_test["gap"]},
        }, step=step)

    # 5) Save and finish
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "two_tower_final.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()
    print("Training complete. Checkpoints saved to:", SAVE_DIR)
