import csv
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, HfArgumentParser

from data import Document, tokenize_query
from indexer import FaissIndexer, NaiveIndexer, load
from models import Encoder, Pooler


@dataclass
class Arguments:
    input_file: str
    document_file: str
    index_file: str
    model: str = "bert-base-uncased"
    max_length: Optional[int] = None
    top_k: int = 10
    batch_size: int = 16
    strict: bool = False
    cache_dir: Optional[str] = None


def load_tsv_data(file) -> Iterable[Document]:
    with open(file, "r") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if i == 0 and row[0] == "id":
                continue
            yield {"id": row[0], "title": row[2], "text": row[1]}  # type: ignore


class Retriever:
    def __init__(self, model: Encoder, tokenizer, document_file, index_file, max_length=None):
        documents = {i: document for i, document in enumerate(load_tsv_data(document_file))}
        indexer = load(index_file)
        print("INDEX:", type(indexer), len(indexer), indexer.dim)
        if isinstance(indexer, NaiveIndexer) and torch.cuda.is_available():
            indexer.index = indexer.index.to(torch.device("cuda"))
        elif isinstance(indexer, FaissIndexer) and torch.cuda.is_available():
            indexer = indexer.to(-1, shard=True)

        self.model = model
        self.tokenizer = tokenizer
        self.documents = documents
        self.indexer = indexer
        self.max_length = max_length

    @torch.no_grad()
    def __call__(
        self, query: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[Document]]]:
        encoding = tokenize_query(
            self.tokenizer, query, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        embeddings = self.model(**encoding)["pooler_output"]
        scores, indices = self.indexer.search(embeddings, k)
        documents = [[self.documents[idx] for idx in idxs] for idxs in indices.cpu().numpy()]
        return scores.tolist(), documents


def main(args: Arguments):
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files={"test": args.input_file}, cache_dir=cache_dir)
    dataset = raw_datasets["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    if hasattr(model, "pooler"):
        model.pooler = Pooler()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model.cuda())
    model.eval()

    retriever = Retriever(model, tokenizer, args.document_file, args.index_file, args.max_length)
    ranks = [-1] * len(dataset)

    batch_size = args.batch_size * max(1, torch.cuda.device_count())
    with tqdm(total=len(dataset)) as pbar:
        for offset in range(0, len(dataset), batch_size):
            batch = dataset[offset : offset + batch_size]
            _, documents = retriever(batch["query"], args.top_k)

            for i, predicted_docs in enumerate(documents):
                if args.strict:
                    correct_doc_id = batch["positive_documents"][i][0]["id"]
                    for j, doc in enumerate(predicted_docs):
                        if doc["id"] == correct_doc_id:
                            ranks[offset + i] = j
                            break
                    continue

                index = _find_match(batch["answers"][i], predicted_docs)
                if index is not None:
                    ranks[offset + i] = index

            pbar.update(len(batch["query"]))

    for eval_rank in (1, 5, 10, 20, 50, 100):
        if eval_rank > args.top_k:
            break
        recall = sum(r < eval_rank for r in ranks if r > -1) / len(ranks)
        print(f"R@{eval_rank}: {recall}")


def _find_match(answers: List[str], documents: List[Document]) -> Optional[int]:
    def _normalize(s: str) -> str:
        return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", s.lower()))

    answers = [_normalize(answer) for answer in answers]
    for i, text in enumerate(_normalize(doc["text"]) for doc in documents):
        for answer in answers:
            if answer in text:
                return i

    return None


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
