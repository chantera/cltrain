import argparse
import os
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"

_safetensor_available = False


try:
    import safetensors.torch

    _safetensor_available = True
except ImportError:
    pass


def extract_encoder(
    type_: str,
    source: Union[str, os.PathLike],
    destination: Union[str, os.PathLike],
    model_name: Optional[str] = None,
):
    if type_ == "query":
        prefix = "query_encoder."
    elif type_ == "entry":
        prefix = "entry_encoder."
    else:
        raise ValueError(f"`type_` must be either 'query' or 'entry', got '{type_}'")

    def _extract(state_dict, prefix):
        new_state_dict = {}
        prefix_len = len(prefix)
        for key, param in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[prefix_len:]] = param
        return new_state_dict

    is_safetensor_model = (Path(source) / SAFE_WEIGHTS_NAME).exists()

    if is_safetensor_model:
        if not _safetensor_available:
            raise ImportError("safetensors is not available")
        state_dict = safetensors.torch.load_file(Path(source) / SAFE_WEIGHTS_NAME, device="cpu")
    else:
        if not (Path(source) / WEIGHTS_NAME).exists():
            raise FileNotFoundError(
                f"neither '{WEIGHTS_NAME}' nor '{SAFE_WEIGHTS_NAME}' found in '{source}'"
            )
        state_dict = torch.load(Path(source) / WEIGHTS_NAME, map_location="cpu")

    new_state_dict = _extract(state_dict, prefix)
    if not new_state_dict:
        raise ValueError(f"No parameters found with prefix '{prefix}'")

    if not Path(destination).exists():
        Path(destination).mkdir()

    if model_name is not None:
        _save_new_model(model_name, new_state_dict, destination, is_safetensor_model)
    elif is_safetensor_model:
        safetensors.torch.save_file(new_state_dict, Path(destination) / SAFE_WEIGHTS_NAME)
    else:
        torch.save(new_state_dict, Path(destination) / WEIGHTS_NAME)


def _save_new_model(model_name, state_dict, dest, safe_serialization=False):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(None, config=config, state_dict=state_dict)

    config.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    model.save_pretrained(dest)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_extract = subparsers.add_parser("extract")
    parser_extract.add_argument("src", type=str)
    parser_extract.add_argument("dest", type=str)
    parser_extract.add_argument("--type", type=str, choices=["query", "entry"], required=True)
    parser_extract.add_argument("--model", type=str)

    args = parser.parse_args()

    if args.command == "extract":
        extract_encoder(args.type, args.src, args.dest, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
