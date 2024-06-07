import torch


def ddp_forward_fuse_batch(model, inputs):
    outputs = model(**dict(inputs, labels=None))

    # NOTE: gather not only entry embeddings but also query embeddings with labels
    # and compute the loss for all queries across devices. This is because outputs
    # of `torch.distributed.all_gather` do not hold grad_fn and thus the gradients
    # on the local query embeddings for all entries and the gradients on the local
    # entry embeddings for all queries must be computed on each device.
    query_embeddings, num_queries = _gather(outputs["query_embeddings"])
    entry_embeddings, num_entries = _gather(outputs["entry_embeddings"])
    labels = _shift_labels(_gather(inputs["labels"])[0], num_queries, num_entries)
    scores = _unwrap_model(model).sim(query_embeddings, entry_embeddings)
    loss = _unwrap_model(model).loss(scores, labels)

    # retain scores only for the local queries.
    with torch.no_grad():
        i = torch.distributed.get_rank()
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
    return outputs


def ddp_forward_no_fuse_batch(model, inputs):
    return model(**inputs)


def dp_forward_fuse_batch(model, inputs):
    outputs = model(**dict(inputs, labels=None))
    scores = _unwrap_model(model).sim(outputs["query_embeddings"], outputs["entry_embeddings"])
    loss = _unwrap_model(model).loss(scores, inputs["labels"])
    outputs.update({"loss": loss, "logits": scores})
    return outputs


def dp_forward_no_fuse_batch(model, inputs):
    chunks = torch.arange(len(inputs["entry_input_ids"])).chunk(len(model.device_ids))
    diff = len(chunks[0]) - len(chunks[-1])
    if diff == 0:
        return model(**inputs)  # no need to adjust entries

    # insert additional entries to have the same number of entries on each device.
    inputs = inputs.copy()
    for k in ["entry_input_ids", "entry_attention_mask"]:
        inputs[k] = torch.cat([inputs[k], inputs[k][:diff]])
    outputs = model(**inputs)
    outputs["entry_embeddings"] = outputs["entry_embeddings"][:-diff]
    return outputs


def _unwrap_model(model):
    return _unwrap_model(model.module) if hasattr(model, "module") else model


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
