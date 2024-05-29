import json
import os
from typing import Any, Dict, Iterable, Union


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))

    basename = os.path.basename(args.input)
    with open(args.output, mode="w", encoding="utf-8") as f:
        for i, entry in enumerate(read_nq(args.input)):
            entry["id"] = f"{basename}-{i}"
            f.write(encoder.encode(entry))
            f.write("\n")


def read_nq(
    file: Union[str, bytes, os.PathLike], use_hard_negative: bool = True
) -> Iterable[Dict[str, Any]]:
    def _convert(entry):
        return {"id": entry["passage_id"], "title": entry["title"], "text": entry["text"]}

    with open(file, "r") as f:
        data = json.load(f)
    for item in data:
        negative_ctxs = item.get("hard_negative_ctxs", []) if use_hard_negative else []
        if not negative_ctxs:
            negative_ctxs = item.get("negative_ctxs", [])
        yield {
            "query": item["question"],
            "answers": item["answers"],
            "positive_documents": [_convert(entry) for entry in item["positive_ctxs"]],
            "negative_documents": [_convert(entry) for entry in negative_ctxs],
        }


if __name__ == "__main__":
    main()
