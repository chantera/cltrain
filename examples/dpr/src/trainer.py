from dataclasses import dataclass

import torch
import transformers
from transformers.training_args import ParallelMode


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    fuse_batch: bool = False


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)
        self._fuse_batch = getattr(self.args, "fuse_batch", False)
        self._evaluator = _Evaluator(self.args.per_device_eval_batch_size * 4)
        self.data_collator = self._wrap_collator(self.data_collator)

    def compute_loss(self, model, inputs, return_outputs=False):
        if model.training:
            outputs = self._compute_loss(model, inputs)
        else:
            outputs = self._compute_loss(model, dict(inputs, output_embeddings=True))
            self._store(inputs, outputs)

        loss = outputs["loss"]

        if return_outputs:
            return (loss, {"loss": outputs["loss"], "logits": outputs["logits"]})
        else:
            return loss

    def _compute_loss(self, model, inputs):
        def _forward(model, inputs):
            return super(self.__class__, self).compute_loss(model, inputs, return_outputs=True)[1]

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self._fuse_batch:
            outputs = _forward(model, dict(inputs, encode_only=True))
            # NOTE: gather not only document embeddings but also query embeddings with labels
            # and compute the loss for all queries across devices. This is because outputs of
            # `torch.distributed.all_gather` do not hold grad_fn and thus the gradients on the
            # local query embeddings for all documents and the gradients on the local document
            # embeddings for all queries must be computed on each device.
            query_embeddings, num_queries = _gather(outputs["query_embeddings"])
            document_embeddings, num_documents = _gather(outputs["document_embeddings"])
            labels = _shift_labels(_gather(inputs["labels"])[0], num_queries, num_documents)
            scores = query_embeddings @ document_embeddings.T
            loss = torch.nn.functional.cross_entropy(scores, labels)
            # retain scores only for the local queries.
            with torch.no_grad():
                i = self.args.process_index
                q_ofs = sum(num_queries[:i])
                d_ofs = sum(num_documents[:i])
                scores = scores[q_ofs : q_ofs + num_queries[i]]  # local queries
                scores = torch.cat(
                    [  # put the local documents first
                        scores[:, d_ofs : d_ofs + num_documents[i]],
                        scores[:, :d_ofs],
                        scores[:, d_ofs + num_documents[i] :],
                    ],
                    dim=1,
                )
            outputs.update({"loss": loss, "logits": scores})
            outputs["_"] = (query_embeddings, document_embeddings, labels)  # cache
        elif self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            if self._fuse_batch:
                outputs = _forward(model, dict(inputs, encode_only=True))
                scores = outputs["query_embeddings"] @ outputs["document_embeddings"].T
                loss = torch.nn.functional.cross_entropy(scores, inputs["labels"])
                outputs.update({"loss": loss, "logits": scores})
            else:
                # insert additional documents to have the same number of documents on each device.
                chunks = torch.arange(len(inputs["document_input_ids"])).chunk(self.args.n_gpu)
                diff = len(chunks[0]) - len(chunks[-1])
                if diff > 0:
                    inputs = inputs.copy()
                    for k in ["document_input_ids", "document_attention_mask"]:
                        inputs[k] = torch.cat([inputs[k], inputs[k][:diff]])
                outputs = _forward(model, inputs)
                if diff > 0 and outputs["document_embeddings"] is not None:
                    outputs["document_embeddings"] = outputs["document_embeddings"][:-diff]
        else:
            outputs = _forward(model, inputs)

        return outputs

    def _wrap_collator(self, data_collator):
        class DataParallelCollator:
            def __init__(self, collate_fn, n_gpu):
                self.collate_fn = collate_fn
                self.n_gpu = n_gpu

            def __call__(self, batch):
                encoding = self.collate_fn(batch)
                # NOTE: assume that `torch.nn.DataParallel.scatter` uses `tensor.chunk` internally.
                # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/csrc/cuda/comm.cpp#L335
                labels = []
                chunks = torch.arange(len(encoding["document_input_ids"])).chunk(self.n_gpu)
                for chunk, ids in zip(chunks, encoding["labels"].chunk(self.n_gpu)):
                    in_chunk = torch.logical_and(chunk[0] <= ids, ids <= chunk[-1])
                    ids = torch.where(in_chunk, ids - chunk[0], -100)
                    labels.append(ids)
                encoding["labels"] = torch.cat(labels)
                return encoding

        if self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED and not self._fuse_batch:
            return DataParallelCollator(data_collator, self.args.n_gpu)
        return data_collator

    def _store(self, inputs, outputs):
        query_embeddings = outputs["query_embeddings"]
        document_embeddings = outputs["document_embeddings"]
        labels = inputs["labels"]

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            if self._fuse_batch:
                query_embeddings, document_embeddings, labels = outputs.pop("_")
            else:
                query_embeddings, num_queries = _gather(query_embeddings, dst=0)
                document_embeddings, num_documents = _gather(document_embeddings, dst=0)
                labels, num_labels = _gather(labels, dst=0)
                if self.args.process_index != 0:
                    return
                assert all(torch.tensor(num_labels) == torch.tensor(num_queries))
                labels = _shift_labels(labels, num_queries, num_documents)
        elif self.args.parallel_mode == ParallelMode.NOT_DISTRIBUTED and not self._fuse_batch:
            query_chunks = torch.arange(len(query_embeddings)).chunk(self.args.n_gpu)
            num_queries = [len(chunk) for chunk in query_chunks]
            document_chunks = torch.arange(len(document_embeddings)).chunk(self.args.n_gpu)
            num_documents = [len(chunk) for chunk in document_chunks]
            labels = _shift_labels(labels, num_queries, num_documents)

        if self.args.process_index == 0:
            self._evaluator.add(query_embeddings, document_embeddings, labels)

    def _compute_metrics(self, p: transformers.EvalPrediction):
        average_rank = -1
        if self.args.process_index == 0:
            average_rank = self._evaluator.evaluate()
            self._evaluator.reset()

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(axis=1)
        accuracy = (preds == p.label_ids).astype("float").mean().item()

        return {"accuracy": accuracy, "rank": average_rank}


class _Evaluator:
    def __init__(self, batch_size=8, device=None):
        self.reset()
        self.batch_size = batch_size
        self.device = device

    def add(self, query_embeddings, document_embeddings, labels):
        if self.device:
            query_embeddings = query_embeddings.to(self.device)
            document_embeddings = document_embeddings.to(self.device)
            labels = labels.to(self.device)

        assert len(labels) == len(query_embeddings)
        offset = len(self.document_embeddings) if self.document_embeddings is not None else 0
        labels = labels + offset
        if self.labels is None:
            self.labels = labels
        else:
            self.labels = torch.cat([self.labels, labels])

        if self.query_embeddings is None:
            self.query_embeddings = query_embeddings
        else:
            self.query_embeddings = torch.cat([self.query_embeddings, query_embeddings])

        if self.document_embeddings is None:
            self.document_embeddings = document_embeddings
        else:
            self.document_embeddings = torch.cat([self.document_embeddings, document_embeddings])

    def evaluate(self):
        total_rank = 0
        document_embs_transposed = self.document_embeddings.T

        for i in range(0, len(self.query_embeddings), self.batch_size):
            query_embs = self.query_embeddings[i : i + self.batch_size]
            labels = self.labels[i : i + self.batch_size]
            scores = query_embs @ document_embs_transposed
            scores, doc_idxs = scores.sort(descending=True)
            query_idxs, ranks = (doc_idxs == labels[:, None]).nonzero().T
            assert len(query_idxs) == len(query_embs)
            assert all(idx == i for i, idx in enumerate(query_idxs.tolist()))
            total_rank += ranks.sum().item()

        return total_rank / len(self.query_embeddings)

    def reset(self):
        self.query_embeddings = None
        self.document_embeddings = None
        self.labels = None


@torch.no_grad()
def _shift_labels(labels, num_queries, num_documents):
    offsets = []
    ofs = 0
    for nq, nd in zip(num_queries, num_documents):
        offsets.extend([ofs] * nq)
        ofs += nd
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
