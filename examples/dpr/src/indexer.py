import os
from typing import Any, List, Optional, Tuple, Union

import torch

_faiss_available = False


try:
    import faiss
    import faiss.contrib.torch_utils

    _faiss_available = True
except ImportError:
    pass


class Indexer:
    def add(self, vectors: torch.Tensor) -> None:
        raise NotImplementedError

    def search(self, queries: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class NaiveIndexer(Indexer):
    def __init__(self, dim: int, dtype: Optional[Any] = None):
        self.index = torch.empty((0, dim), dtype=dtype)

    @torch.no_grad()
    def add(self, vectors: torch.Tensor) -> None:
        assert vectors.ndim == 2
        self.index = torch.cat([self.index, vectors])

    @torch.no_grad()
    def search(self, queries: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        assert queries.ndim == 2
        return (queries @ self.index.T).topk(k)

    def reset(self) -> None:
        self.index = self.index.new_empty((0, self.dim))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def dim(self) -> int:
        return self.index.size(1)


class FaissIndexer(Indexer):
    def __init__(self, dim: int):
        if not _faiss_available:
            raise RuntimeError("faiss is not available")
        metric = faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexFlat(dim, metric)
        self._device: Optional[List[int]] = None

    def add(self, vectors: torch.Tensor) -> None:
        self.index.add(self._validate_tensor(vectors))

    def search(self, queries: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.index.search(self._validate_tensor(queries), k)

    def _validate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._device and tensor.is_cuda:
            raise ValueError("GPU tensor cannot be used with CPU index")
        elif self._device and not tensor.is_cuda:
            raise ValueError("CPU tensor cannot be used with GPU index")

        # NOTE: tensor must be placed on CPU to transfer it to multiple GPUs
        if self._device is not None and len(self._device) > 1:
            tensor = tensor.cpu()

        return tensor

    def reset(self) -> None:
        self.index.reset()

    def __len__(self) -> int:
        return self.index.ntotal

    @property
    def dim(self) -> int:
        return self.index.d

    def to(self, device: Optional[Union[int, List[int]]], shard: bool = False) -> "FaissIndexer":
        if device is None:
            device_idxs = None
        elif isinstance(device, int):
            device_idxs = [device] if device >= 0 else list(range(torch.cuda.device_count()))
        else:
            device_idxs = device

        if device_idxs == self._device:
            return self

        index = self.index
        if self._device and device_idxs:  # GPU -> GPU
            co = faiss.GpuMultipleClonerOptions()
            co.shard = shard
            index = faiss.index_gpu_to_cpu(index)
            index = faiss.index_cpu_to_gpus_list(index, co=co, gpus=device_idxs)
        elif self._device and not device_idxs:  # GPU -> CPU
            index = faiss.index_gpu_to_cpu(index)
        elif not self._device and device_idxs:  # CPU -> GPU
            co = faiss.GpuMultipleClonerOptions()
            co.shard = shard
            index = faiss.index_cpu_to_gpus_list(index, co=co, gpus=device_idxs)
        else:  # CPU -> CPU
            raise AssertionError("this cannot be reached")

        indexer = FaissIndexer(self.dim)
        indexer.index = index
        indexer._device = device_idxs
        return indexer


def save(indexer: Indexer, file: Union[str, bytes, os.PathLike]):
    if isinstance(indexer, NaiveIndexer):
        torch.save(indexer.index, file)
    elif isinstance(indexer, FaissIndexer):
        faiss.write_index(indexer.to(device=None).index, file)
    else:
        raise NotImplementedError


def load(file: Union[str, bytes, os.PathLike]) -> Indexer:
    if not os.path.exists(file):
        raise FileNotFoundError(f"No such file or directory: {file!r}")

    if _faiss_available:
        try:
            index = faiss.read_index(file)
            instance: Indexer = FaissIndexer(0)
            instance.index = index  # type: ignore[attr-defined]
            return instance
        except RuntimeError:
            pass

    index = torch.load(file)
    instance = NaiveIndexer(0)
    instance.index = index  # type: ignore[attr-defined]
    return instance
