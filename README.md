# SAGERec

SAGERec is a similarity-aware generalization framework for zero-shot cross-domain sequential recommendation. The model is trained on an interaction-rich source domain and directly transferred to unseen target domains without using target-domain interactions during training. Item-side text is encoded by an LLM encoder and used as transferable semantic information across domains.

The implementation supports four sequential recommendation backbones:

- BERT4Rec
- SASRec
- GRU4Rec
- UniSRec

It also supports three training objectives:

- `sem`: semantic sequential recommendation without cross-domain regularization.
- `recg`: semantic generalization baseline with alignment.
- `sage`: the proposed similarity-aware and domain-adaptive generalization objective.

## Project structure

```text
sagerec_release/
├── README.md
├── requirements.txt
├── src/
│   └── sagerec/
│       ├── __init__.py
│       ├── datasets.py
│       ├── losses.py
│       ├── models.py
│       ├── trainer.py
│       └── utils.py
└── scripts/
    ├── preprocess_amazon.py
    ├── encode_items.py
    └── train.py
```

## Environment

```bash
pip install -r requirements.txt
```

For item-text encoding, install `llm2vec` and the corresponding model dependencies required by your local environment.

## Data format

Each domain is placed under `data/<domain>/`.

For training and evaluation, the required processed file is:

```text
data/<domain>/processed_data.csv
```

Required columns:

```text
UserId, ItemId, Timestamp
```

Item-text columns used for LLM encoding include:

```text
title, description, features
```

For Steam-style data, the encoder also supports:

```text
products, text
```

The generated item semantic embedding file should be:

```text
data/<domain>/<domain>_embedding_llama.parquet
```

with columns:

```text
ItemId, item_text_embedding
```

The model uses `0` as the padding index. Real item ids are mapped to the dense range `1..N` during dataset loading.

## Preprocess Amazon-style datasets

Raw files are expected under each domain directory:

```text
data/<domain>/<domain>.csv
data/<domain>/meta_<domain>.jsonl
```

Run:

```bash
python scripts/preprocess_amazon.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
  --min_interactions 10 \
  --max_seq_len 50
```

The script applies 10-core filtering, aligns interactions with metadata, removes repeated user-item interactions, truncates user histories to the latest 50 interactions, and writes `processed_data.csv`.

## Encode item text with LLM2Vec

```bash
python scripts/encode_items.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
  --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
  --embedding_tag llama \
  --batch_size 256
```

This produces one parquet embedding file for each domain.

## Train SAGERec

Example: train on Movies & TV and evaluate zero-shot transfer to CDs & Vinyl and Steam.

```bash
python scripts/train.py \
  --data_root ./data \
  --source_domain amazon_movies_and_tv \
  --all_domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
  --target_domains amazon_cds_and_vinyl,steam \
  --model bert4rec \
  --loss_mode sage \
  --epochs 100 \
  --batch_size 128 \
  --hidden_units 256 \
  --max_len 50 \
  --num_heads 2 \
  --num_layers 2 \
  --dropout_rate 0.2 \
  --learning_rate 1e-4 \
  --lambda_g 0.05 \
  --gamma_g 0.1 \
  --beta_id 0.1 \
  --tau 0.2 \
  --sim_threshold 0.0
```

Results are saved to:

```text
outputs/results.csv
```

The checkpoint is saved under:

```text
outputs/checkpoints/
```

## Run different backbones

Use the `--model` option:

```bash
--model sasrec
--model bert4rec
--model gru4rec
--model unisrec
```

For UniSRec, the number of MoE experts can be set by:

```bash
--n_exps 8
```

## Run different objectives

```bash
--loss_mode sem
--loss_mode recg
--loss_mode sage
```

A typical comparison can be launched as:

```bash
for model in sasrec bert4rec gru4rec unisrec
do
  for loss in sem recg sage
  do
    python scripts/train.py \
      --data_root ./data \
      --source_domain amazon_movies_and_tv \
      --all_domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
      --target_domains amazon_cds_and_vinyl,steam \
      --model $model \
      --loss_mode $loss \
      --epochs 100 \
      --batch_size 128
  done
done
```

## Evaluation protocol

For each target domain, the model is evaluated in a zero-shot manner. The target-domain interaction sequences are used only for evaluation, not for parameter optimization. The default evaluation uses sampled negative items and reports Recall and NDCG at the configured top-k values.

Default top-k values:

```bash
--topk 10,20
```

Default number of sampled negatives:

```bash
--eval_negatives 100
```

## Main hyperparameters

| Argument | Description | Default |
|---|---|---:|
| `--hidden_units` | Hidden dimension | 256 |
| `--max_len` | Maximum sequence length | 50 |
| `--num_heads` | Transformer attention heads | 2 |
| `--num_layers` | Sequential encoder layers | 2 |
| `--dropout_rate` | Dropout rate | 0.2 |
| `--batch_size` | Training batch size | 128 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--lambda_g` | Base semantic regularization strength | 0.05 |
| `--gamma_g` | Domain-adaptive decay coefficient | 0.1 |
| `--beta_id` | Intra-domain diversity weight | 0.1 |
| `--tau` | Temperature for semantic regularization | 0.2 |
| `--sim_threshold` | Similarity threshold for cross-domain item alignment | 0.0 |

## Citation

Please cite the corresponding SAGERec paper when using this code.
