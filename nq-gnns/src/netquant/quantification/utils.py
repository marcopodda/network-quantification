import torch
from quapy.data import LabelledCollection


def to_quapy_collection(
    graph_embeddings: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> LabelledCollection:
    instances = graph_embeddings[mask].detach().to("cpu").numpy()
    labels = labels[mask].detach().to("cpu").numpy()
    return LabelledCollection(instances, labels)


def needs_calibration(method: str) -> bool:
    return method in ["ACC", "PACC", "HDy", "DMTopsoe"]
