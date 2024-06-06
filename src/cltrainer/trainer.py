from dataclasses import dataclass

import transformers
from transformers.training_args import ParallelMode

from .data import DataParallelCollator
from .models import ModelForContrastiveLearning
from .parallel import (
    ddp_forward_fuse_batch,
    ddp_forward_no_fuse_batch,
    dp_forward_fuse_batch,
    dp_forward_no_fuse_batch,
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    fuse_batch: bool = False


class ContrastiveLearningTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, ModelForContrastiveLearning):
            raise TypeError(f"model must be an instance of {ModelForContrastiveLearning}")

        self._fuse_batch = getattr(self.args, "fuse_batch", False)
        self.data_collator = self._wrap_collator(self.data_collator)

    def compute_loss(self, model, inputs, return_outputs=False):
        parallel_mode = self.args.parallel_mode
        if parallel_mode == ParallelMode.DISTRIBUTED:
            forward_fn = ddp_forward_fuse_batch if self._fuse_batch else ddp_forward_no_fuse_batch
        elif parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            forward_fn = dp_forward_fuse_batch if self._fuse_batch else dp_forward_no_fuse_batch
        elif parallel_mode == ParallelMode.NOT_PARALLEL:
            forward_fn = lambda model, inputs: model(**inputs)  # noqa
        else:
            raise NotImplementedError(f"{parallel_mode=} is not supported")

        outputs = forward_fn(model, inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _wrap_collator(self, data_collator):
        if self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            return DataParallelCollator(data_collator, self.args.n_gpu, self._fuse_batch)
        return data_collator
