from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union

import torch
from transformers import PreTrainedTokenizerBase


class Document(TypedDict):
    id: str
    text: str
    title: Optional[str]


class Entry(TypedDict):
    query: str
    positive_documents: List[Document]
    negative_documents: List[Document]


class BatchEntry(TypedDict):
    query: List[str]
    positive_documents: List[List[Document]]
    negative_documents: List[List[Document]]


class Preprocessor:
    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        document_tokenizer: PreTrainedTokenizerBase,
        include_title: bool = True,
        use_negative: bool = True,
        max_length: Optional[int] = None,
    ):
        self.query_tokenizer = query_tokenizer
        self.document_tokenizer = document_tokenizer
        self.include_title = include_title
        self.use_negative = use_negative
        self.max_length = max_length

    def __call__(self, entry: Union[Entry, BatchEntry]) -> Dict[str, Any]:
        if not isinstance(entry["query"], list):
            batch: BatchEntry = {k: [v] for k, v in entry.items()}  # type: ignore
            return {k: v[0] for k, v in self.__call__(batch).items()}

        queries: List[str] = entry["query"]
        documents: List[Document] = []
        offsets = []

        for positive_documents, negative_documents in zip(
            entry["positive_documents"], entry["negative_documents"]
        ):
            offsets.append(len(documents))
            documents.append(positive_documents[0])
            if self.use_negative and len(negative_documents) > 0:
                documents.append(negative_documents[0])
        offsets.append(len(documents))

        query_input_ids = tokenize_query(
            self.query_tokenizer, queries, max_length=self.max_length
        )["input_ids"]
        document_input_ids = tokenize_document(
            self.document_tokenizer, documents, self.include_title, max_length=self.max_length
        )["input_ids"]

        return {
            "query_input_ids": query_input_ids,
            "document_input_ids": [
                document_input_ids[offsets[i] : offsets[i + 1]] for i in range(len(queries))
            ],
            "label": [0] * len(queries),  # index for positive document
        }


def tokenize_query(
    tokenizer: PreTrainedTokenizerBase, batch: List[str], **kwargs
) -> Dict[str, Any]:
    return tokenizer.batch_encode_plus(batch, truncation=True, **kwargs)


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase,
    batch: Iterable[Document],
    include_title: bool = True,
    **kwargs
) -> Dict[str, Any]:
    batch_text: List[Any]
    if include_title:
        batch_text = [(document["title"], document["text"]) for document in batch]
    else:
        batch_text = [document["text"] for document in batch]
    # NOTE: `token_type_ids` is discarded as `title` and `text` are not distinguished.
    return tokenizer.batch_encode_plus(
        batch_text, truncation=True, return_token_type_ids=False, **kwargs
    )


class Collator:
    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        document_tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ):
        self.query_tokenizer = query_tokenizer
        self.document_tokenizer = document_tokenizer
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        query_input_ids = []
        document_input_ids = []
        labels = []
        offset = 0
        for item in batch:
            query_input_ids.append(item["query_input_ids"])
            document_input_ids.extend(item["document_input_ids"])
            labels.append(offset + item["label"])
            offset += len(item["document_input_ids"])

        query_encoding = self._pad(query_input_ids, self.query_tokenizer)
        document_encoding = self._pad(document_input_ids, self.document_tokenizer)
        if self.device:
            query_encoding.to(self.device)
            document_encoding.to(self.device)

        batch_encoding = {
            "query_input_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "document_input_ids": document_encoding["input_ids"],
            "document_attention_mask": document_encoding["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
        }
        return batch_encoding

    @classmethod
    def _pad(cls, encoded_inputs: List[List[int]], tokenizer):
        return tokenizer.pad({"input_ids": encoded_inputs}, padding=True, return_tensors="pt")
