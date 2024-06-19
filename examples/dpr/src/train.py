import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from cltrain import (
    ContrastiveLearningTrainer,
    DataCollatorForContrastiveLearning,
    ModelForContrastiveLearning,
    TrainingArguments,
)
from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, HfArgumentParser, set_seed

from data import Preprocessor
from model_utils import document_model_from_pretrained, query_model_from_pretrained
from training_utils import LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    query_model: str = "bert-base-uncased"
    document_model: str = "bert-base-uncased"
    max_seq_length: Optional[int] = None
    use_negative: bool = False
    cache_dir: Optional[str] = None


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")  # noqa  # fmt: skip
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    query_config = AutoConfig.from_pretrained(args.query_model)
    document_config = AutoConfig.from_pretrained(args.document_model)
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    document_tokenizer = AutoTokenizer.from_pretrained(args.document_model)

    preprocessor = Preprocessor(
        query_tokenizer,
        document_tokenizer,
        max_length=args.max_seq_length,
        use_negative=args.use_negative,
    )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        datasets = raw_datasets.map(preprocessor, batched=True)
        for column_name in next(iter(datasets.values())).column_names:
            if not column_name.startswith("document_"):
                continue
            datasets = datasets.rename_column(column_name, f"entry_{column_name[9:]}")

    query_encoder = query_model_from_pretrained(args.query_model, config=query_config)
    document_encoder = document_model_from_pretrained(args.document_model, config=document_config)
    model = ModelForContrastiveLearning(query_encoder, document_encoder)

    trainer = ContrastiveLearningTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        data_collator=DataCollatorForContrastiveLearning(query_tokenizer, document_tokenizer),
        compute_metrics=compute_metrics,
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
            if training_args.should_save:
                output_dir = Path(training_args.output_dir)
                query_tokenizer.save_pretrained(output_dir / "query")
                document_tokenizer.save_pretrained(output_dir / "document")
                query_encoder.save_pretrained(output_dir / "query")
                document_encoder.save_pretrained(output_dir / "document")
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
        result = trainer.predict(datasets["test"])
        logger.info(f"test metrics: {result.metrics}")
        trainer.log_metrics("predict", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("predict", result.metrics)


def compute_metrics(p: EvalPrediction):
    logits, query_embs, document_embs = p.predictions
    preds = logits.argmax(axis=1)
    targets = p.label_ids != -100
    accuracy = (preds[targets] == p.label_ids[targets]).astype("float").mean().item()

    num_documents_per_query = len(document_embs) // len(query_embs)
    labels = torch.arange(0, len(document_embs), num_documents_per_query).numpy()
    scores = query_embs @ document_embs.T
    ranks = ((-scores).argsort() == labels[:, None]).nonzero()[1] + 1
    assert len(ranks) == len(query_embs)
    average_rank = ranks.mean()
    mean_reciprocal_rank = (1 / ranks).mean()

    return {"accuracy": accuracy, "average_rank": average_rank, "mrr": mean_reciprocal_rank}


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "training.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    main(args, training_args)
