# Two-Tower Retrieval Model

This project implements a Two-Tower neural retrieval model for the MS MARCO dataset. Queries and their associated positive passages are encoded separately, and training uses a custom triplet loss with in-batch negatives.

---

## 🗂 Directory Structure

```
.
├── checkpoints/                 # Saved model checkpoints
├── conda_env.yml               # Environment setup with Mamba
├── data/                       # Parquet files, GloVe embeddings, vocab
├── .env                        # WandB API key (not tracked)
├── misc/                       # Debug and exploration scripts
├── notebooks/                  # Dataset download, token updates, tests
├── scratch/                    # Intermediate dev work (not used in final pipeline)
├── scripts/
│   ├── config.py               # Global configs for trianing 
│   ├── dataloader.py           # Loads queries, positive passages, and samples negatives
│   └── train.py                # Full training loop, model, and metrics
└── structure.txt               # Tree structure of the repo
```

---

## 📦 Setup

1. **Install with Mamba:**
```bash
mamba env create -f conda_env.yml
mamba activate two_tower_env
```

2. **Set up WandB API key**  
   Create a `.env` file:
```
WANDB_API_KEY=your_key_here
```

---

## 📋 Dataset Info

We use the MS MARCO v2.1 dataset, downloaded via Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("ms_marco", "v2.1")
```

We save the splits as parquet in `data/`.

| Column                  | Type               | Notes                                                |
| -----------------------|-------------------|------------------------------------------------------|
| `query`                | string            | Raw query                                            |
| `passages.passage_text`| list of strings   | 10 candidate passages                                |
| `passages.is_selected` | list of ints      | Labels (not used; all 10 are treated as positive)   |

---

## 🔡 Vocab + Embeddings

- Pretrained GloVe embeddings: `glove.6B.200d.txt`
- We added `<pad>` (zero vector) and `<unk>` (mean embedding) tokens to the end.
- Process done in `notebooks/add_tokens.ipynb`.

---

## 🧠 Model Overview

Implemented in `scripts/train.py`:

- **Encoders**: Two separate RNNs for queries and passages.
- **Loss**: Custom `full_batch_triplet_loss`
  - Each query is paired with 10 positive passages.
  - Each positive is matched with one negative sampled from other queries.
- **Metrics**: `recall@1`, `recall@5`, and `gap`

---

## 🧪 Dataloader Details

In `scripts/dataloader.py`:

- Each batch contains:
  - `B` queries
  - `B x 10` positive passages
- Negatives are drawn dynamically during training from other examples in the batch.

---

## 🚀 Training

Run:

```bash
python scripts/train.py
```

Logs to [Weights & Biases](https://wandb.ai/) if `.env` is set.

Final model is saved to `checkpoints/two_tower_final.pt`.

---

## 📝 Notebooks

- `download_dataset.ipynb`: Fetch MS MARCO from Hugging Face
- `add_tokens.ipynb`: Add `<pad>` and `<unk>` to vocab + embeddings
- `two_tower_test_working_sample.ipynb`: Basic end-to-end test

---

## ✅ Status

- ✅ Full support for multiple positives per query
- ✅ Dynamic in-batch negative sampling
- ✅ Works with GloVe-based initialisation
- ✅ Logging to wandb

---

## 📚 Citation

Dataset:  
> MS MARCO: https://huggingface.co/datasets/microsoft/ms_marco