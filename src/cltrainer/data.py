from typing import Any, Dict, List, Optional

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class DataCollatorForContrastiveLearning:
    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        entry_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.query_tokenizer = query_tokenizer
        self.entry_tokenizer = entry_tokenizer

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Each example consists of one query and one or more entries.
        This assumes that `example["query_*"]` is `Any` and `example["entry_*"]` is `List[Any]`.
        """
        query_inputs = {k[6:]: [] for k in examples[0].keys() if k.startswith("query_")}  # type: ignore
        entry_inputs = {k[6:]: [] for k in examples[0].keys() if k.startswith("entry_")}  # type: ignore
        labels = []
        offset = 0

        for example in examples:
            for k, v in example.items():
                if k.startswith("query_"):
                    query_inputs[k[6:]].append(v)
                elif k.startswith("entry_"):
                    entry_inputs[k[6:]].extend(v)
            label = example["label"]
            labels.append(offset + label if label != -100 else label)
            offset += len(example["entry_input_ids"])

        query_tokenizer = self.query_tokenizer
        entry_tokenizer = self.entry_tokenizer
        if entry_tokenizer is None:
            entry_tokenizer = query_tokenizer

        query_batch = query_tokenizer.pad(query_inputs, padding=True, return_tensors="pt")
        entry_batch = entry_tokenizer.pad(entry_inputs, padding=True, return_tensors="pt")
        batch = BatchEncoding(
            {
                **{"query_" + k: v for k, v in query_batch.items()},
                **{"entry_" + k: v for k, v in entry_batch.items()},
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
        return batch
