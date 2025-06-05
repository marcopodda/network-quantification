from collections import Counter

import numpy as np
import torch
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator, ClassifierMixin
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from nq.models.utils import to_quapy_collection
from nq.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ENQ(BaseEstimator, ClassifierMixin):
    def __init__(self, radius: int = 2) -> None:
        self.radius = radius
        self.ego_networks_ = {}
        self.labels_ = None
        self.classes_ = None
        self.train_mask_ = None
        self.prevalences_ = None
        self.num_classes_ = None
        self.graph_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, graph: Data, train_mask: torch.Tensor) -> None:
        # compute ego network for each node in the network
        self.graph_ = graph

        log.info("Extracting ego networks...")
        edge_index = graph.edge_index.to(self.device)

        for i, center_node in enumerate(range(graph.num_nodes)):
            subset, _, _, _ = k_hop_subgraph(
                node_idx=center_node,
                num_hops=self.radius,
                edge_index=edge_index,
                relabel_nodes=False,
            )
            self.ego_networks_[i] = subset.cpu().numpy().tolist()
        log.info("Extracted ego networks.")

        self.labels_ = to_quapy_collection(graph.x, graph.y).y
        self.classes_ = np.unique(self.labels_)
        self.train_mask_ = train_mask

        train_data = to_quapy_collection(graph.x, graph.y, train_mask)
        self.prevalences_ = train_data.prevalence()
        self.num_classes_ = train_data.classes_
        self.classes_ = np.unique(self.labels_)

    def predict(self, X) -> np.ndarray:
        predictions = []

        for instance in X:
            if self.train_mask_[instance]:
                predictions.append(self.labels_[instance])
            else:
                ego_network_nodes = self.ego_networks_[instance]
                node_label_frequencies = Counter([self.labels_[node_id] for node_id in ego_network_nodes if self.train_mask_[node_id]])
                if len(node_label_frequencies) == 0:
                    predictions.append(np.random.choice(self.num_classes_, p=self.prevalences_))
                else:
                    most_common_label = node_label_frequencies.most_common(1)[0][0]
                    predictions.append(most_common_label)

        return np.array(predictions)

    def to_quapy_data(self, fold_index):
        quapy_data = {}

        for stage in ["train", "val", "cal", "test"]:
            mask = self.graph_[stage + "_mask"]

            instances = np.arange(self.graph_.num_nodes)[mask[fold_index]]
            labels = self.graph_.y.int()[mask[fold_index]].numpy()
            quapy_data[stage] = LabelledCollection(instances, labels)

        return quapy_data
