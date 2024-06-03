from dataclasses import dataclass

import torch
import transformers
from transformers.modeling_utils import unwrap_model
from transformers.training_args import ParallelMode

from cltrainer.models import ModelForContrastiveLearning


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
        def _forward(model, inputs, return_loss=True):
            if not return_loss:
                inputs = dict(inputs, labels=None)
            compute_loss = super(ContrastiveLearningTrainer, self).compute_loss
            return compute_loss(model, inputs, return_outputs=True)[1]

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self._fuse_batch:
            outputs = _forward(model, inputs, return_loss=False)
            # NOTE: gather not only entry embeddings but also query embeddings with labels
            # and compute the loss for all queries across devices. This is because outputs of
            # `torch.distributed.all_gather` do not hold grad_fn and thus the gradients on the
            # local query embeddings for all entries and the gradients on the local entry
            # embeddings for all queries must be computed on each device.
            query_embeddings, num_queries = _gather(outputs["query_embeddings"])
            entry_embeddings, num_entries = _gather(outputs["entry_embeddings"])
            labels = _shift_labels(_gather(inputs["labels"])[0], num_queries, num_entries)
            scores = unwrap_model(model).sim(query_embeddings, entry_embeddings)
            loss = unwrap_model(model).loss(scores, labels)

            # retain scores only for the local queries.
            with torch.no_grad():
                i = self.args.process_index
                q_ofs = sum(num_queries[:i])
                e_ofs = sum(num_entries[:i])
                scores = scores[q_ofs : q_ofs + num_queries[i]]  # local queries
                scores = torch.cat(
                    [  # put the local entries first
                        scores[:, e_ofs : e_ofs + num_entries[i]],
                        scores[:, :e_ofs],
                        scores[:, e_ofs + num_entries[i] :],
                    ],
                    dim=1,
                )

            outputs.update({"loss": loss, "logits": scores})
        elif self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            if self._fuse_batch:
                outputs = _forward(model, inputs, return_loss=False)
                scores = model.module.sim(outputs["query_embeddings"], outputs["entry_embeddings"])
                loss = model.module.loss(scores, inputs["labels"])
                outputs.update({"loss": loss, "logits": scores})
            else:
                # insert additional entries to have the same number of entries on each device.
                chunks = torch.arange(len(inputs["entry_input_ids"])).chunk(self.args.n_gpu)
                diff = len(chunks[0]) - len(chunks[-1])
                if diff > 0:
                    inputs = inputs.copy()
                    for k in ["entry_input_ids", "entry_attention_mask"]:
                        inputs[k] = torch.cat([inputs[k], inputs[k][:diff]])
                outputs = _forward(model, inputs)
                if diff > 0:
                    outputs["entry_embeddings"] = outputs["entry_embeddings"][:-diff]
        else:
            outputs = _forward(model, inputs)

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _wrap_collator(self, data_collator):
        class _DataParallelCollator:
            def __init__(self, collate_fn, n_gpu):
                self.collate_fn = collate_fn
                self.n_gpu = n_gpu

            def __call__(self, batch):
                encoding = self.collate_fn(batch)
                # NOTE: assume that `torch.nn.DataParallel.scatter` uses `tensor.chunk` internally.
                # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/csrc/cuda/comm.cpp#L335
                labels = []
                chunks = torch.arange(len(encoding["entry_input_ids"])).chunk(self.n_gpu)
                for chunk, ids in zip(chunks, encoding["labels"].chunk(self.n_gpu)):
                    in_chunk = torch.logical_and(chunk[0] <= ids, ids <= chunk[-1])
                    ids = torch.where(in_chunk, ids - chunk[0], -100)
                    labels.append(ids)
                encoding["labels"] = torch.cat(labels)
                return encoding

        if self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED and not self._fuse_batch:
            return _DataParallelCollator(data_collator, self.args.n_gpu)
        return data_collator


@torch.no_grad()
def _shift_labels(labels, num_queries, num_entries):
    offsets = []
    ofs = 0
    for nq, ne in zip(num_queries, num_entries):
        offsets.extend([ofs] * nq)
        ofs += ne
    return torch.where(labels >= 0, labels + labels.new_tensor(offsets), labels)


@torch.no_grad()
def _gather_shape(tensor):
    shape = torch.tensor(tensor.shape, device=tensor.device)[None]
    buf = [torch.empty_like(shape) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(buf, shape)
    return torch.cat(buf)


def _gather(tensor, dst=None):
    lengths = _gather_shape(tensor).T[0].tolist()
    if len(tensor) < max(lengths):
        pad = [0, 0] * tensor.ndim
        pad[-1] = max(lengths) - len(tensor)
        tensor = torch.nn.functional.pad(tensor, pad)

    buf = None
    if dst is None or dst == torch.distributed.get_rank():
        buf = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]

    if dst is None:
        torch.distributed.all_gather(buf, tensor.contiguous())
    else:
        torch.distributed.gather(tensor.contiguous(), buf, dst=dst)

    out = None
    if buf is not None:
        buf[torch.distributed.get_rank()] = tensor
        out = torch.cat([v[:n] for v, n in zip(buf, lengths)])

    return out, lengths
