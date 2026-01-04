# CLAUDE.md

## Project Overview

BSARec (Beyond Self-Attention for Sequential Recommendation) - AAAI 2024 paper implementation combining Fourier transform-based frequency filtering with self-attention for sequential recommendation.

## Repository Structure

```
src/
├── main.py           # Entry point for training/evaluation
├── trainers.py       # Trainer class with train/valid/test loops
├── dataset.py        # RecDataset and data loading utilities
├── utils.py          # Logging, argument parsing, EarlyStopping
├── metrics.py        # recall_at_k, ndcg_k evaluation metrics
├── model/
│   ├── __init__.py   # MODEL_DICT registry
│   ├── _abstract_model.py  # SequentialRecModel base class
│   ├── _modules.py   # LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock
│   ├── bsarec.py     # Main model (FrequencyLayer + Attention)
│   └── [baselines]   # sasrec, bert4rec, fmlprec, gru4rec, caser, duorec, fearec
├── data/             # Datasets: Beauty, ML-1M, Yelp, Sports, Toys, LastFM
└── output/           # Checkpoints (.pt) and logs (.log)
```

## Dataset

### Data Source

- Amazon reviews (Beauty, Sports, Toys) from SNAP Stanford
- MovieLens-1M from GroupLens
- LastFM from GroupLens hetrec2011

### Data Pre-processing

- Parses JSON/CSV formats into (user, item, timestamp) tuples
- Filters by rating score (removes low ratings)
- Applies 5-core filtering (users and items must have ≥5 interactions)
- Maps raw IDs to sequential integers starting from 1
- Final output: is a text file, stored in `../DatasetName.txt` format, where each line is:
  - `user_id item_id_1 item_id_2 item_id_3 ...`
    - Items are ordered chronologically by interaction timestamp.

## Commands

```bash
# Training
cd src
python main.py --data_name Beauty --model_type BSARec --lr 0.0005 --alpha 0.7 --c 5 --train_name BSARec_Beauty

# Evaluation
python main.py --data_name Beauty --model_type BSARec --load_model BSARec_Beauty_best --do_eval

# Environment setup
conda env create -f bsarec_env.yaml && conda activate bsarec
```

## Key Hyperparameters

- `--alpha`: Frequency vs attention weight (BSARec)
- `--c`: Low-pass filter cutoff (BSARec)
- `--hidden_size`: Embedding dim (default: 64)
- `--num_hidden_layers`: Transformer blocks (default: 2)
- `--max_seq_length`: Sequence length (default: 50)

## Adding New Models

1. Create `src/model/newmodel.py` inheriting from `SequentialRecModel`
2. Implement `forward()` and `calculate_loss()` methods
3. Register in `MODEL_DICT` in `src/model/__init__.py`
