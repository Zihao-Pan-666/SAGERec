# SAGERec

Implementation for **SAGERec: Similarity-Aware Generalization Framework for Zero-Shot Cross-Domain Sequential Recommendation**.

SAGERec aims to improve zero-shot cross-domain sequential recommendation. The model is trained on a source domain and directly evaluated on unseen target domains. It uses LLM-based item semantic representations and a similarity-aware generalization objective to support transferable user behavior modeling across domains.

The current implementation supports the following sequential recommendation backbones:


- BERT4Rec
- SASRec
- GRU4Rec
- UniSRec

It also supports three training objectives:

- `sem`: semantic sequential recommendation baseline
- `recg`: semantic generalization baseline
- `sage`: the proposed similarity-aware generalization objective

---

## Requirements

The experiments were conducted with Python 3.9+ and PyTorch 2.6.0. The local environment used the CUDA 12.6 build of PyTorch (`torch==2.6.0+cu126`). Please install the PyTorch version that matches your CUDA environment.

Key packages:

- `torch==2.6.0`
- `transformers==4.40.2`
- `datasets==3.1.0`
- `evaluate==0.4.3`
- `llm2vec==0.2.2`
- `peft==0.11.1`
- `accelerate==1.12.0`
- `huggingface-hub==0.36.0 `
- `tokenizers==0.19.1 `
- `safetensors==0.7.0`
- `numpy==2.0.2`
- `pandas==2.2.3`
- `pyarrow==23.0.0`
- `scikit-learn==1.5.2`
- `scipy==1.13.1`
- `tqdm==4.66.5`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all scripts from the project root. If needed, set the Python path first:

```bash
export PYTHONPATH=$PWD
```

---

## Structure

```text
SAGERec/
├── sagerec/
│   ├── __init__.py
│   ├── datasets.py          # Dataset loading and sequence construction
│   ├── losses.py            # Training objectives and regularization losses
│   ├── models.py            # SASRec, BERT4Rec, GRU4Rec, and UniSRec backbones
│   ├── trainer.py           # Training and evaluation loops
│   └── utils.py             # Utility functions
├── scripts/
│   ├── preprocess_amazon.py # Amazon-style data preprocessing
│   ├── encode_items.py      # LLM-based item text encoding
│   └── train.py             # Training and zero-shot evaluation
├── README.md
└── requirements.txt
```

---

## Usage

### 1. Data Preparation

The raw Amazon review data used in the experiments can be obtained from **Amazon Reviews 2023**:

```text
https://amazon-reviews-2023.github.io/
```

The dataset is also available on Hugging Face:

```text
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
```

For each domain, download the corresponding review file and metadata file, then place them under `data/{domain}/`.

The current preprocessing script expects the following local file names:

```text
data/{domain}/{domain}.csv
data/{domain}/meta_{domain}.jsonl
```

Run preprocessing:

```bash
python scripts/preprocess_amazon.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
  --min_interactions 10 \
  --max_seq_len 50
```

The processed data will be saved as:

```text
data/{domain}/processed_data.csv
```

The processed file should contain at least:

```text
UserId, ItemId, Timestamp
```

Item text fields such as `title`, `description`, and `features` are used for semantic item encoding when available.

Due to storage and redistribution constraints, processed data files and pre-computed item embeddings are not included in this repository. They can be generated using the provided scripts. If you need the exact processed files used in the paper, please contact the authors.

### 2. Item Semantic Encoding

Generate item semantic embeddings with LLM2Vec. The default model can be downloaded directly from Hugging Face:

```text
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
```

If the model has already been downloaded locally, you can also pass the local model path to `--model_name`, for example:

```text
./llm2vec-llama-3-8B-Instruct-mntp
```

Run item encoding:

```bash
python scripts/encode_items.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam \
  --model_name McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
  --embedding_tag llama \
  --batch_size 256
```

This produces one embedding file for each domain:

```text
data/{domain}/{domain}_embedding_llama.parquet
```

The embedding file should contain:

```text
ItemId, item_text_embedding
```

### 3. Training and Zero-Shot Evaluation

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
  --learning_rate 1e-4
```

Available backbones:

```bash
--model bert4rec
--model sasrec
--model gru4rec
--model unisrec
```

Available objectives:

```bash
--loss_mode sem
--loss_mode recg
--loss_mode sage
```

Evaluation uses sampled negative items and reports Recall and NDCG at the configured top-k values. By default, the script uses:

```bash
--topk 10,20
--eval_negatives 100
```

Results are saved to:

```text
outputs/results.csv
```

Model checkpoints are saved under:

```text
outputs/checkpoints/
```

---

## Citation

If you use this code, please cite the corresponding SAGERec paper.
