# DPR

## Setup

You can create an virtual environment by running the following commands.

```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Usage

Training and inference are performed using [src/train.py](src/train.py).

### Data Preparation

A dataset used for the training script must be a collection of examples for contrastive learning.
Each example consists of the following fields:

- `query` (string): *a query text*
- `positive_documents` (list of documents): *positive documents*
- `negative documents` (list of documents): *negative documents*

Each document has `id` (string), `text` (string), and `title` (string).

For reproduction of DPR, you need to download datasets by following the instructions in [facebookresearch/DPR](https://github.com/facebookresearch/DPR#1-download-all-retriever-training-and-validation-data).
For using `run_training.sh`, convert the datasets and place the outputs in `data` directory.

```sh
$ python data/convert_nq_to_jsonl.py $DPR_DATA/retriever/nq-train.json data/nq-train.jsonl
$ python data/convert_nq_to_jsonl.py $DPR_DATA/retriever/nq-dev.json data/nq-dev.jsonl
```

### Training

```sh
$ ./run_training.sh
```

You can give additional training arguments (e.g., `--save_total_limit 5`, `--seed 42`).

### Indexing

You need to install [facebookresearch/faiss](https://github.com/facebookresearch/faiss).

```sh
$ python3 src/build.py \
    --input_file $DPR_DATA/wikipedia_split/psgs_w100.tsv \
    --output_file $OUTPUT_DIR/psgs_w100.index \
    --model $OUTPUT_DIR/document \
    --max_length 256
```

### Evaluation

```sh
$ python src/retrieve.py \
    --input_file data/nq-dev.jsonl \
    --document_file $DPR_DATA/wikipedia_split/psgs_w100.tsv \
    --index_file $OUTPUT_DIR/psgs_w100.index \
    --model $OUTPUT_DIR/query \
    --top_k 100
```
