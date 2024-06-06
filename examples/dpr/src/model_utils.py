import torch
from transformers import AutoModel, DPRConfig, DPRContextEncoder, DPRQuestionEncoder


class Pooler(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states[:, 0]


def query_model_from_pretrained(
    pretrained_model_name_or_path, config=None, replace_pooler=True, **kwargs
):
    model_cls = DPRQuestionEncoder if isinstance(config, DPRConfig) else AutoModel
    model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
    if replace_pooler and hasattr(model, "pooler"):
        model.pooler = Pooler()
    return model


def document_model_from_pretrained(
    pretrained_model_name_or_path, config=None, replace_pooler=True, **kwargs
):
    model_cls = DPRContextEncoder if isinstance(config, DPRConfig) else AutoModel
    model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
    if replace_pooler and hasattr(model, "pooler"):
        model.pooler = Pooler()
    return model
