# Path-LLM: Official Implementation

## Overview

**Path-LLM** is a framework for graph representation learning using large language models (LLMs). It provides tools for:
- L2SP selection for Path-LLM,
- Path-LLM self-supervised pre-training on L2SP-based paths,
- Extracting node embeddings from fine-tuned Path-LLM,
- Evaluating Path-LLM embeddings on node classification, edge validation and keyword search tasks.

## Directory Structure

```
main code/
  ├── SP_processing.py
  ├── PathLLM_tuning.py
  ├── no_tuning_pathllm.py
  ├── nc.py
  ├── ev.py
  ├── calculate_newgraph_weight.py
  └── keyword_search_case.py
```

## Requirements

- Python 3.9+
- torch 2.3.0
- transformers 4.45.1
- trl 0.8.6
- peft 0.11.1
- datasets 2.19.2
- scikit-learn 1.4.2
- pandas 2.23
- numpy 1.26.4
- scipy 1.13.1
- networkx 3.2.1
- spacy 3.8.0
- wandb 0.19.9
- nltk 3.8.1

Install dependencies (example):
```bash
pip install torch transformers trl peft datasets scikit-learn pandas numpy scipy networkx spacy pytextrank wandb matplotlib nltk
python -m spacy download en_core_web_sm
```

## Data

The code expects datasets in folders such as `PubMed`, `Freebase`, `Cora`, `Citeseer`, `ARXIV`, and `Facebook`, with files like:
- `Cora/nc_dataset/label.dat.test`
- `Cora/nc_dataset/label.dat.train`



**Note:** You may need to prepare or download these datasets separately.

## Main Scripts

### 1. SP_processing.py

Select L2SP paths in graphs.

**Usage:**
- Edit the script to set the mode (`PubMed`, `Freebase`, `Cora`, `Citeseer`, `ARXIV`, or `Facebook`).
- Run:
  ```bash
  python SP_processing.py
  ```

### 2. PathLLM_tuning.py

Fine-tune Path-LLM through the self-supervised learning on L2SP-based paths.

**Usage:**
- Place your HuggingFace access token and training data in `training_data.txt`.
- Edit the script to set your access token.
- Run:
  ```bash
  python PathLLM_tuning.py
  ```

### 3. no_tuning_pathllm.py

Extract node embeddings from the fine-tuned Path-LLM.

**Usage:**
```bash
python no_tuning_pathllm.py <model_path> <output_embedding_path>
```
- `model_path`: Path to Path-LLM.
- `output_embedding_path`: File to save node embeddings.

### 4. nc.py

Evaluate node classification performance.

**Usage:**
```bash
python nc.py <embedding_file> <result_name>
```

### 5. lp.py

Evaluate edge validation performance.

**Usage:**
```bash
python ev.py <embedding_file> <result_name>
```


### 6. calculate_newgraph_weight.py

Calculate edge weights for a graph based on node embeddings derived by Path-LLM.

**Usage:**
- Edit the script to set the input and output file paths.
- Run:
  ```bash
  python calculate_newgraph_weight.py
  ```

### 7. keyword_search_case.py

Case study for keyword-based path search.

**Usage:**
- Edit the script to set the input file paths.
- Run:
  ```bash
  python keyword_search_case.py
  ```

## Notes

- You need to adjust file paths and device settings in the scripts.
- For Path-LLM fine-tuning, ensure you have the appropriate HuggingFace access and model weights.