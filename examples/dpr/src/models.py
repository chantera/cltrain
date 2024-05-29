from typing import Any, Dict, Optional, Protocol, TypedDict

import torch


class Encoder(Protocol):
    class Output(TypedDict):
        pooler_output: torch.Tensor

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Output:
        ...


class BiEncoder(torch.nn.Module):
    def __init__(self, query_encoder: Encoder, document_encoder: Encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output["pooler_output"]

    def encode_document(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.document_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output["pooler_output"]

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        document_input_ids: torch.Tensor,
        document_attention_mask: torch.Tensor,
        encode_only: bool = False,
        output_embeddings: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        query_embs = self.encode_query(query_input_ids, query_attention_mask)
        document_embs = self.encode_document(document_input_ids, document_attention_mask)

        output_embeddings = output_embeddings or encode_only
        output = {
            "loss": None,
            "logits": None,
            "query_embeddings": query_embs if output_embeddings else None,
            "document_embeddings": document_embs if output_embeddings else None,
        }
        if encode_only:
            return output

        scores = query_embs @ document_embs.T
        loss = torch.nn.functional.cross_entropy(scores, labels) if labels is not None else None
        output.update({"loss": loss, "logits": scores})
        return output


class Pooler(torch.nn.Module):
    def __init__(self, target_index: int = 0):
        super().__init__()
        self.target_index = target_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, self.target_index]
