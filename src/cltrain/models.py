from typing import Callable, Optional, Protocol, TypedDict

import torch
import torch.nn.functional as F


class Encoder(Protocol):
    class Output(TypedDict):
        last_hidden_state: torch.Tensor
        pooler_output: torch.Tensor

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Output: ...


class ModelForContrastiveLearning(torch.nn.Module):
    class Output(TypedDict):
        loss: Optional[torch.Tensor]
        logits: Optional[torch.Tensor]
        query_embeddings: torch.Tensor
        entry_embeddings: torch.Tensor

    def __init__(
        self,
        query_encoder: Encoder,
        entry_encoder: Optional[Encoder] = None,
        query_pooler: Optional[Callable[[Encoder.Output], torch.Tensor]] = None,
        entry_pooler: Optional[Callable[[Encoder.Output], torch.Tensor]] = None,
        sim: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.query_pooler = query_pooler
        self.entry_encoder = entry_encoder
        self.entry_pooler = entry_pooler

        if sim is None:
            if temperature is not None:
                sim = CosineSimilarity(temperature)
            else:
                sim = DotProductSimilarity()
        self.sim = sim
        self.loss = torch.nn.CrossEntropyLoss()

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        output = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if self.query_pooler is not None:
            return self.query_pooler(output)
        return output["pooler_output"]

    def encode_entry(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        entry_encoder = self.entry_encoder
        entry_pooler = self.entry_pooler
        if entry_encoder is None:
            entry_encoder = self.query_encoder
            entry_pooler = self.query_pooler

        output = entry_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if entry_pooler is not None:
            return entry_pooler(output)
        return output["pooler_output"]

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        entry_input_ids: torch.Tensor,
        entry_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Output:
        query_embs = self.encode_query(query_input_ids, query_attention_mask)
        entry_embs = self.encode_entry(entry_input_ids, entry_attention_mask)

        output: ModelForContrastiveLearning.Output = {
            "loss": None,
            "logits": None,
            "query_embeddings": query_embs,
            "entry_embeddings": entry_embs,
        }
        if labels is None:
            return output

        scores = self.sim(query_embs, entry_embs)
        loss = self.loss(scores, labels)
        output.update({"loss": loss, "logits": scores})
        return output


class DotProductSimilarity(torch.nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 @ x2.T


class CosineSimilarity(torch.nn.Module):
    __constants__ = ["temp"]
    temp: float

    def __init__(self, temp: float = 1.0) -> None:
        super().__init__()
        self.temp = temp

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1) / self.temp

    def extra_repr(self) -> str:
        return f"temp={self.temp}"
