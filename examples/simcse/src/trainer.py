import sys

import torch
from cltrainer import ContrastiveLearningTrainer
from transformers import EvalPrediction

PATH_TO_SENTEVAL = "./SentEval"
PATH_TO_DATA = PATH_TO_SENTEVAL + "/data"

_senteval_available = False


try:
    if PATH_TO_SENTEVAL not in sys.path:
        sys.path.insert(0, PATH_TO_SENTEVAL)
    import senteval

    _senteval_available = True
except ImportError:
    pass


class Trainer(ContrastiveLearningTrainer):
    def __init__(self, *args, **kwargs):
        do_sts_eval = kwargs.pop("do_sts_eval", False)
        if do_sts_eval and not _senteval_available:
            raise RuntimeError("senteval is not available")

        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)

        self.do_sts_eval = do_sts_eval

    def _compute_metrics(self, p: EvalPrediction):
        logits, embs1, embs2 = p.predictions
        preds = logits.argmax(axis=1)
        targets = p.label_ids != -100
        accuracy = (preds[targets] == p.label_ids[targets]).astype("float").mean().item()

        embs1 = torch.nn.functional.normalize(torch.from_numpy(embs1), p=2, dim=-1)
        embs2 = torch.nn.functional.normalize(torch.from_numpy(embs2), p=2, dim=-1)
        align_loss_val = align_loss(embs1[targets], embs2[targets]).item()
        uniform_loss_val1 = uniform_loss(embs1).item()
        uniform_loss_val2 = uniform_loss(embs2).item()

        metrics = {
            "accuracy": accuracy,
            "align_loss": align_loss_val,
            "uniform_loss": (uniform_loss_val1 + uniform_loss_val2) / 2,
        }

        if self.do_sts_eval:
            metrics.update(evaluate_sts(self.model, self.tokenizer))

        return metrics


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


# https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/trainers.py#L93-L144
def evaluate_sts(model, tokenizer):
    device = next(model.parameters()).device
    model.eval()

    def prepare(params, samples):
        return

    @torch.no_grad()
    def batcher(params, batch):
        sentences = [" ".join(s) for s in batch]
        batch = tokenizer(sentences, return_tensors="pt", padding=True)
        return model.encode_query(**batch.to(device)).cpu()

    params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
    params["classifier"] = {
        "nhid": 0,
        "optim": "rmsprop",
        "batch_size": 128,
        "tenacity": 3,
        "epoch_size": 2,
    }

    se = senteval.engine.SE(params, batcher, prepare)
    results = se.eval(["STSBenchmark", "SICKRelatedness"])

    stsb_spearman = results["STSBenchmark"]["spearman"]
    sickr_spearman = results["SICKRelatedness"]["spearman"]

    metrics = {
        "stsb_spearman": stsb_spearman,
        "sickr_spearman": sickr_spearman,
        "avg_sts": (stsb_spearman + sickr_spearman) / 2,
    }
    return metrics
