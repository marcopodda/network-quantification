from collections import Counter

import demon as d
import numpy as np
import torch
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator, ClassifierMixin
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import to_networkx

from nq.models.utils import to_quapy_collection


class CDQ(BaseEstimator, ClassifierMixin):
    def __init__(self, merge_strategy: str = "frequency_based") -> None:
        self.merge_strategy = merge_strategy
        self.graph_ = None
        self.communities_ = None
        self.predictions_ = None
        self.prevalences_ = None
        self.num_classes_ = None
        self.labels_ = None
        self.classes_ = None
        self.train_mask = None

    def fit(self, graph: Data, train_mask: torch.Tensor):
        # compute communities
        self.graph_ = graph
        nx_graph = to_networkx(graph, node_attrs=["x"])
        dm = d.Demon(graph=nx_graph, epsilon=0.25, min_community_size=3)
        self.communities_ = dm.execute()

        # compute prevalences train set
        train_data = to_quapy_collection(graph.x, graph.y, train_mask)
        self.prevalences_ = train_data.prevalence()
        self.num_classes_ = train_data.classes_
        self.labels_ = graph.y.numpy()
        self.classes_ = np.unique(self.labels_)
        self.train_mask_ = train_mask

        # precompute classification for each node
        predictions = {}
        for community in self.communities_:
            node_label_frequencies = Counter([self.labels_[node_id] for node_id in community if self.train_mask_[node_id]])
            most_common_label = node_label_frequencies.most_common(1)
            if len(most_common_label) > 0:
                if self.merge_strategy == "frequency_based":
                    for node_id in community:
                        if node_id in predictions and predictions[node_id][1] < most_common_label[0][1] / len(community):
                            frequency = most_common_label[0][1] / len(community)
                            predictions[node_id] = (most_common_label[0][0], frequency)
                        elif node_id not in predictions:
                            frequency = most_common_label[0][1] / len(community)
                            predictions[node_id] = (most_common_label[0][0], frequency)
                elif self.merge_strategy == "density_based":
                    edge_index_community, _ = subgraph(community, graph.edge_index)
                    community_density = edge_index_community.shape[1] / (len(community) * (len(community) - 1)) // 2
                    for node_id in community:
                        if node_id in predictions and predictions[node_id][1] < community_density:
                            predictions[node_id] = (most_common_label[0][0], community_density)
                        elif node_id not in predictions:
                            predictions[node_id] = (most_common_label[0][0], community_density)
                else:
                    raise Exception(f"Merge strategy {self.merge_strategy} not implemented")

        self.predictions_ = {key: value[0].item() for key, value in predictions.items()}

    def predict(self, X) -> np.ndarray:
        instances_predictions = []

        for instance in X:
            if self.train_mask_[instance]:
                instances_predictions.append(self.labels_[instance])
            elif instance in self.predictions_:
                instances_predictions.append(self.predictions_[instance])
            else:
                instances_predictions.append(np.random.choice(self.num_classes_, p=self.prevalences_))
        return np.array(instances_predictions)

    def to_quapy_data(self, fold_index):
        quapy_data = {}

        for stage in ["train", "val", "cal", "test"]:
            mask = self.graph_[stage + "_mask"]

            instances = np.arange(self.graph_.num_nodes)[mask[fold_index]]
            labels = self.graph_.y.int()[mask[fold_index]].numpy()
            quapy_data[stage] = LabelledCollection(instances, labels)

        return quapy_data
