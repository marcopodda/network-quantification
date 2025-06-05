import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

import netquant
from netquant.modules.module import NodeClassificationModule
from netquant.quantification import DATASETS
from netquant.quantification.utils import to_quapy_collection


def load_graph(name: str) -> Data:
    return DATASETS[name]()[0]


def load_fold_data(embeddings: torch.Tensor, graph: Data, fold_index: int):
    train_data = to_quapy_collection(embeddings, graph.y, graph.train_mask[fold_index])
    calibration_data = to_quapy_collection(embeddings, graph.y, graph.cal_mask[fold_index])
    validation_data = to_quapy_collection(embeddings, graph.y, graph.val_mask[fold_index])
    test_data = to_quapy_collection(embeddings, graph.y, graph.test_mask[fold_index])
    return train_data, calibration_data, validation_data, test_data


def load_model(cfg: DictConfig) -> NodeClassificationModule:
    root = netquant.EXP_ROOT / cfg.dataset_name / cfg.model_name / "train"
    root = root / f"fold_{cfg.fold_index}" / f"L={cfg.num_layers}-h={cfg.hidden_dim}"
    ckpts = (root / "checkpoints").glob("epoch*.ckpt")
    return NodeClassificationModule.load_from_checkpoint(next(ckpts), map_location="cpu")
