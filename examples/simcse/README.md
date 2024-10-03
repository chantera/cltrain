# SimCSE

## Setup

Prerequisite: You need to checkout [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE).

You can create an virtual environment by running the following commands.

```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ ln -s $SIMCSE/SentEval 
```

For validation and evaluation, you also need to prepare datasets in `$SIMCSE/SentEval/data`.
Please follow the instruction in [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE/blob/main/README.md#evaluation).

## Usage

Training and inference are performed using [src/train.py](src/train.py).

### Data Preparation

A dataset used for the training script must be a collection of examples for contrastive learning.
Examples in the dataset must be in one of the following forms:

- `string`: a single text (for unsupervised learning)
- `(string, string)`: a tuple of a text and the corresponding positive text (for supervised learning with positive examples)
- `(string, string, string)`: a triple of a text, the positive text, and the negative text (for supervised learning with positive and negative examples)

A dataset can be provided as a CSV file, where the first, second, and third columns represent a text, positive text, and negative text, respectively.
You can omit the second and third columns when you do not use positive and negative texts.
For unsupervised learning, you can also use a text file, where each line represents a single text.

For reproduction of SimCSE, you need to download datasets in `$SIMCSE/data`.
Please follow the instructions in [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE/blob/main/README.md#training).
For using `run_unsup_training.sh` and `run_sup_training.sh`, place the datasets in `data` directory.

```sh
$ mkdir data
$ cp $SIMCSE/data/wiki1m_for_simcse.txt ./data/  # For unsupervised training
$ cp $SIMCSE/data/nli_for_simcse.csv ./data/  # For supervised training
# Instead of copying, you can create a symbolic link by `ln -s $SIMCSE/data`.
```

### Training

```sh
$ ./run_unsup_training.sh  # Use `run_sup_training.sh` for supervised training.
```

You can give additional training arguments (e.g., `--save_total_limit 5`, `--seed 42`).

### Evaluation

```sh
$ python $SIMCSE/evaluation.py \
    --model_name_or_path ./output/my-unsup-simcse-bert-base-uncased/encoder \
    --pooler cls_before_pooler \  # Use `cls` for the supervised SimCSE setting
    --task_set sts \
    --mode test
```
