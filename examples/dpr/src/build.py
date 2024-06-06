import csv
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import torch
from data import Document, tokenize_document
from indexer import FaissIndexer, Indexer, NaiveIndexer, save
from model_utils import document_model_from_pretrained
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, HfArgumentParser


@dataclass
class Arguments:
    input_file: str
    output_file: str
    model: str = "bert-base-uncased"
    max_seq_length: Optional[int] = None
    batch_size: int = 16
    no_faiss: bool = False


def load_tsv_data(file) -> Iterable[Document]:
    with open(file, "r") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if i == 0 and row[0] == "id":
                continue
            yield {"id": row[0], "title": row[2], "text": row[1]}


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, file, tokenizer, max_length=None, batch_size=1000):
        self.file = file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("multi-process loading is not supported.")

        def _iter_batch(file, batch_size):
            batch = []
            for example in load_tsv_data(file):
                batch.append(example)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        tokenizer = self.tokenizer
        for batch in _iter_batch(self.file, self.batch_size):
            encoding = tokenize_document(
                tokenizer, batch, include_title=True, max_length=self.max_length, truncation=True
            )
            for i in range(len(batch)):
                yield {k: v[i] for k, v in encoding.items()}


def main(args: Arguments):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_loader = torch.utils.data.DataLoader(
        Dataset(args.input_file, tokenizer, args.max_seq_length),
        batch_size=args.batch_size * max(1, torch.cuda.device_count()),
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    config = AutoConfig.from_pretrained(args.model)
    model = document_model_from_pretrained(args.model, config=config)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model.cuda())
    model.eval()

    indexer: Indexer
    if args.no_faiss:
        indexer = NaiveIndexer(config.hidden_size)
    else:
        indexer = FaissIndexer(config.hidden_size)

    with tqdm() as pbar:
        for inputs in data_loader:
            indexer.add(model(**inputs).pooler_output.cpu())
            pbar.update(len(inputs["input_ids"]))

    save(indexer, args.output_file)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
