import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DPRConfig,
    DPRContextEncoder,
    DPRQuestionEncoder,
    HfArgumentParser,
    set_seed,
)

from data import Collator, Preprocessor
from models import BiEncoder, Pooler
from trainer import Trainer, TrainingArguments
from training_utils import LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    query_model: str = "bert-base-uncased"
    document_model: str = "bert-base-uncased"
    max_length: Optional[int] = None
    use_negative: bool = False
    cache_dir: Optional[str] = None
    develop: bool = False


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")

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
        max_length=args.max_length,
        use_negative=args.use_negative,
    )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocessor, batched=True, remove_columns=column_names)

    set_seed(training_args.seed)

    if args.develop:
        scale = 0.25
        for split in list(splits.keys()):
            splits[split] = splits[split].select(range(int(len(splits[split]) * scale)))
        configs = {"query": query_config, "document": document_config}
        for k, config in configs.items():
            configs[k] = DPRConfig(
                hidden_size=int(config.hidden_size * scale),
                num_hidden_layers=int(config.num_hidden_layers * scale),
                num_attention_heads=int(config.num_attention_heads * scale),
                intermediate_size=int(config.intermediate_size * scale),
            )
        query_encoder = DPRQuestionEncoder(configs["query"])
        document_encoder = DPRContextEncoder(configs["document"])
    else:
        query_model_cls = DPRQuestionEncoder if isinstance(query_config, DPRConfig) else AutoModel
        query_encoder = query_model_cls.from_pretrained(args.query_model, config=query_config)
        if hasattr(query_encoder, "pooler"):
            query_encoder.pooler = Pooler()
        document_model_cls = (
            DPRContextEncoder if isinstance(document_config, DPRConfig) else AutoModel
        )
        document_encoder = document_model_cls.from_pretrained(
            args.document_model, config=document_config
        )
        if hasattr(document_encoder, "pooler"):
            document_encoder.pooler = Pooler()

    model = BiEncoder(query_encoder, document_encoder)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits.get("train"),
        eval_dataset=splits.get("validation"),
        data_collator=Collator(query_tokenizer, document_tokenizer),
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
        result = trainer.predict(splits["test"])
        logger.info(f"test metrics: {result.metrics}")
        trainer.log_metrics("predict", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("predict", result.metrics)


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    if args.validation_file is None:
        training_args.evaluation_strategy = "no"
    main(args, training_args)
