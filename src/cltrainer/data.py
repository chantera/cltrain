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


class DataParallelCollator:
    def __init__(self, collate_fn, n_gpu, fuse_batch=False):
        self.collate_fn = collate_fn
        self.n_gpu = n_gpu
        self.fuse_batch = fuse_batch

    def __call__(self, examples):
        batch = self.collate_fn(examples)
        if self.fuse_batch:
            return batch  # no need to adjust labels

        # NOTE: assume that `torch.nn.DataParallel.scatter` uses `tensor.chunk` internally.
        # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/csrc/cuda/comm.cpp#L335
        labels = []
        chunks = torch.arange(len(batch["entry_input_ids"])).chunk(self.n_gpu)
        for chunk, ids in zip(chunks, batch["labels"].chunk(self.n_gpu)):
            in_chunk = torch.logical_and(chunk[0] <= ids, ids <= chunk[-1])
            ids = torch.where(in_chunk, ids - chunk[0], -100)
            labels.append(ids)
        batch["labels"] = torch.cat(labels)
        return batch
