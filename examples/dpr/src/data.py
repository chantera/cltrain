from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union

from transformers import PreTrainedTokenizerBase


class Document(TypedDict):
    id: str
    text: str
    title: Optional[str]


class Example(TypedDict):
    query: str
    positive_documents: List[Document]
    negative_documents: List[Document]


class BatchExample(TypedDict):
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

    def __call__(self, examples: Union[Example, BatchExample]) -> Dict[str, Any]:
        if not isinstance(examples["query"], list):
            examples: BatchExample = {k: [v] for k, v in examples.items()}  # type: ignore
            return {k: v[0] for k, v in self.__call__(examples).items()}

        queries: List[str] = examples["query"]
        documents: List[Document] = []
        offsets = []

        for positive_documents, negative_documents in zip(
            examples["positive_documents"], examples["negative_documents"]
        ):
            # use only the first positive/negative document
            offsets.append(len(documents))
            documents.append(positive_documents[0])
            if self.use_negative and len(negative_documents) > 0:
                documents.append(negative_documents[0])
        offsets.append(len(documents))

        query_features = self.query_tokenizer(queries, max_length=self.max_length, truncation=True)
        document_features = tokenize_document(
            self.document_tokenizer,
            documents,
            self.include_title,
            max_length=self.max_length,
            truncation=True,
        )
        for k, v in document_features.items():
            document_features[k] = [v[offsets[i] : offsets[i + 1]] for i in range(len(queries))]

        batch = {
            **{"query_" + k: v for k, v in query_features.items()},
            **{"document_" + k: v for k, v in document_features.items()},
            "label": [0] * len(queries),  # positive document is the first document in the list
        }
        return batch


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase,
    documents: Iterable[Document],
    include_title: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    batch_text: List[Any]
    if include_title:
        batch_text = [(document["title"], document["text"]) for document in documents]
    else:
        batch_text = [document["text"] for document in documents]
    # NOTE: `token_type_ids` is discarded as `title` and `text` are not distinguished.
    return tokenizer.batch_encode_plus(batch_text, return_token_type_ids=False, **kwargs)
