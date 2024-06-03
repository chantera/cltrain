import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from cltrainer import DataCollatorForContrastiveLearning, ModelForContrastiveLearning
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from trainer import Trainer
from training_utils import LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    eval_dataset: Optional[str] = field(default="mteb/stsbenchmark-sts", metadata={"nargs": "?"})
    model: str = "bert-base-uncased"
    temperature: float = 0.05
    max_length: Optional[int] = None
    sts_eval: bool = False
    cache_dir: Optional[str] = None


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    raw_datasets = load_dataset("text", data_files=data_files, cache_dir=args.cache_dir)

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    preprocessor = Preprocessor(
        tokenizer, max_length=args.max_length, text_column_name="text", prefix="query_"
    )

    def preprocess(examples):
        features = preprocessor(examples)
        for k in list(features.keys()):
            if k.startswith("query_"):
                features["entry_" + k[6:]] = [[v] for v in features[k]]
        features["label"] = [0] * len(features["query_input_ids"])
        return features

    with training_args.main_process_first(desc="dataset map pre-processing"):
        datasets = raw_datasets.map(preprocess, batched=True)

    if args.eval_dataset is not None:
        raw_eval_dataset = load_dataset(
            args.eval_dataset, cache_dir=args.cache_dir, split="validation"
        )

        eval_preprocessor1 = Preprocessor(
            tokenizer, max_length=args.max_length, text_column_name="sentence1", prefix="query_"
        )
        eval_preprocessor2 = Preprocessor(
            tokenizer, max_length=args.max_length, text_column_name="sentence2", prefix="entry_"
        )

        def eval_preprocess(examples):
            features = {**eval_preprocessor1(examples), **eval_preprocessor2(examples)}
            for k in list(features.keys()):
                if k.startswith("entry_"):
                    features[k] = [[v] for v in features[k]]
            features["label"] = [0 if s >= 4.0 else -100 for s in examples["score"]]
            return features

        with training_args.main_process_first(desc="eval_dataset map pre-processing"):
            eval_dataset = raw_eval_dataset.map(eval_preprocess, batched=True)
    else:
        eval_dataset = None

    class Pooler(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]

    encoder = AutoModel.from_pretrained(args.model, config=config)
    model = ModelForContrastiveLearning(
        query_encoder=encoder, query_pooler=Pooler(), temperature=args.temperature
    )
    print(model)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForContrastiveLearning(tokenizer),
        do_sts_eval=args.sts_eval,
    )
    trainer.add_callback(LoggerCallback(logger))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info(f"train metrics: {result.metrics}")
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info(f"eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        pass  # do nothing


class Preprocessor:
    def __init__(
        self,
        tokenizer,
        text_column_name: str = "text",
        prefix: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
        self.prefix = prefix
        self.max_length = max_length

    def __call__(self, examples):
        features = self.tokenizer(
            examples[self.text_column_name], max_length=self.max_length, truncation=True
        )
        if self.prefix is not None:
            features = {f"{self.prefix}{k}": v for k, v in features.items()}
        return features


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "training.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    main(args, training_args)
