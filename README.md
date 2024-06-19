# CL-train: Training utilities for Contrastive Learning

This package provides training utilities for contrastive learning.

## Installation

```
git clone https://github.com/chantera/cltrain.git
cd cltrain
pip install .
```

## Usage

### Model definition

```py
from cltrain import ModelForContrastiveLearning
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
model = ModelForContrastiveLearning(encoder)
```

You can use different encoders for query and entry (a.k.a dual-encoder or bi-encoder).

```py
from cltrain import ModelForContrastiveLearning
from transformers import AutoModel

query_encoder = AutoModel.from_pretrained("bert-base-uncased")
entry_encoder = AutoModel.from_pretrained("bert-base-uncased")
model = ModelForContrastiveLearning(query_encoder, entry_encoder)
```

### Training with in-batch negatives

`ModelForContrastiveLearning` can be trained by giving query and entry features with labels that indicate the indices of the positive examples within the batch entries.

```py
query_features = {f"query_{k}": v in k, v in tokenizer([x["text1"] for x in examples], return_tensors="pt").items()}
entry_features = {f"entry_{k}": v in k, v in tokenizer([x["text2"] for x in examples], return_tensors="pt").items()}
labels = torch.arange(len(examples))

output = model(**query_features, **entry_features, labels=labels)
loss = output["loss"]
```

Training can be done with `transformers.Trainer` or `cltrain.ContrastiveLearningTrainer`.
`ContrastiveLearningTrainer` will accommodate batch fusion for data parallel training to increase in-batch negative examples.

```py
# from transformers import Trainer
from cltrain import ContrastiveLearningTrainer as Trainer, TrainingArguments

training_args = TrainingArguments(fuse_batch=True, **kwargs)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

## Example

The following code conducts contrastive learning using text pairs obtained from an NLI dataset.

```py
from cltrain import ContrastiveLearningTrainer, DataCollatorForContrastiveLearning, ModelForContrastiveLearning, TrainingArguments
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

dataset = load_dataset("stanfordnlp/snli")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(examples):
    """
    Each resulting example consists of one query and one or more entries.
    This assumes that `example["query_*"]` is `Any` and `example["entry_*"]` is `List[Any]`.
    """
    query_features = tokenizer(examples["premise"], truncation=True)
    entry_features = tokenizer(examples["hypothesis"], truncation=True)

    batch = {}
    batch_size = len(examples["premise"])
    for k, v in query_features.items():
        batch[f"query_{k}"] = v
    for k, v in entry_features.items():
        batch[f"entry_{k}"] = [[v[i]] for i in range(batch_size)]
    # You can add negative entries with `batch[f"entry_{k}"][i].extend(values)`

    # Indicate the index of positive entry within the entry list
    batch["label"] = [0] * batch_size  # in this case, the positive entry is always the first element in the list
    return batch

dataset = dataset.filter(lambda x: x["label"] == 0)  # only use entailment text pairs
dataset = dataset.map(preprocess, batched=True)

encoder = AutoModel.from_pretrained("bert-base-uncased")
model = ModelForContrastiveLearning(encoder)

training_args = TrainingArguments(fuse_batch=True, output_dir="./output")
trainer = ContrastiveLearningTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorForContrastiveLearning(tokenizer),
)
trainer.train()
```