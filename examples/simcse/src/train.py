import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from cltrain import DataCollatorForContrastiveLearning, ModelForContrastiveLearning
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser, set_seed

from trainer import Trainer, TrainingArguments
from training_utils import LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    eval_dataset: Optional[str] = field(default="mteb/stsbenchmark-sts", metadata={"nargs": "?"})
    model: str = "bert-base-uncased"
    temperature: float = 0.05
    max_seq_length: Optional[int] = None
    sts_eval: bool = False
    mlp_only_training: bool = False
    cache_dir: Optional[str] = None


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")  # noqa  # fmt: skip
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = None
    if args.train_file is not None:
        raw_train_dataset = load_dataset(
            "csv" if args.train_file.endswith(".csv") else "text",
            data_files={"train": args.train_file},
            cache_dir=args.cache_dir,
            split="train",
        )

        with training_args.main_process_first(desc="train_dataset map pre-processing"):
            column_names = raw_train_dataset.column_names
            train_dataset = raw_train_dataset.map(
                partial(
                    preprocess,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    truncation=True,
                    text_column_name=column_names[0],
                    positive_text_column_name=column_names[1] if len(column_names) > 1 else None,
                    negative_text_column_name=column_names[2] if len(column_names) > 2 else None,
                ),
                batched=True,
            )

    eval_dataset = None
    if args.eval_dataset is not None:
        raw_eval_dataset = load_dataset(
            args.eval_dataset, cache_dir=args.cache_dir, split="validation"
        )

        with training_args.main_process_first(desc="eval_dataset map pre-processing"):
            eval_dataset = raw_eval_dataset.map(
                partial(
                    preprocess,
                    tokenizer=tokenizer,
                    max_length=args.max_seq_length,
                    truncation=True,
                    text_column_name="sentence1",
                    positive_text_column_name="sentence2",
                ),
                batched=True,
            )
            eval_dataset = eval_dataset.map(
                lambda example: {"label": example["label"] if example["score"] >= 4.0 else -100}
            )

    class Pooler(torch.nn.Module):
        # identical to `transformers.modeling_bert.BertPooler` except for training_only option
        def __init__(self, config, training_only=False):
            super().__init__()
            self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = torch.nn.Tanh()
            self.training_only = training_only

        def forward(self, hidden_states):
            pooled_output = hidden_states[:, 0]
            if self.training or not self.training_only:
                pooled_output = self.activation(self.dense(pooled_output))
            return pooled_output

    encoder = AutoModel.from_pretrained(args.model, config=config)
    assert hasattr(encoder, "pooler")
    encoder.pooler = Pooler(config, training_only=args.mlp_only_training)
    model = ModelForContrastiveLearning(encoder, temperature=args.temperature)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
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


def preprocess(
    examples,
    tokenizer,
    text_column_name: str = "text",
    positive_text_column_name: Optional[str] = None,
    negative_text_column_name: Optional[str] = None,
    **kwargs,
):
    query_features = tokenizer(examples[text_column_name], **kwargs)

    positive_entry_features = query_features
    if positive_text_column_name is not None:
        positive_entry_features = tokenizer(examples[positive_text_column_name], **kwargs)

    negative_entry_features = None
    if negative_text_column_name is not None:
        negative_entry_features = tokenizer(examples[negative_text_column_name], **kwargs)

    batch_size = len(examples[text_column_name])
    batch = {f"query_{k}": v for k, v in query_features.items()}

    # each element in an entry feature is a list of feature values for multiple entries
    for k, v in positive_entry_features.items():
        batch[f"entry_{k}"] = [[v[i]] for i in range(batch_size)]

    if negative_entry_features is not None:
        for k, v in negative_entry_features.items():
            for i in range(batch_size):
                batch[f"entry_{k}"][i].append(v[i])

    batch["label"] = [0] * batch_size  # positive entry is always the first entry in the list
    return batch


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "training.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    main(args, training_args)
