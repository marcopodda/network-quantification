from typing import Callable

import numpy as np
import torch
from nq import settings
from nq.datasets.base import BaseDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class Toloker(BaseDataset):
    r"""Load toloker dataset with custom splits."""

    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"

    def __init__(
        self,
        root: str = settings.DATA_DIR.as_posix(),
        name: str = "toloker",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.name}s.npz"]

    def process(self) -> None:
        raw = np.load(self.raw_paths[0], "r")

        # node features
        x = torch.from_numpy(raw["node_features"]).float()

        # node labels
        y = torch.from_numpy(raw["node_labels"]).long()
        y = torch.where(y < 0, 0, y).squeeze().float()

        # adjacency matrix
        edge_index = torch.from_numpy(raw["edges"]).t().long()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0)).contiguous()

        # create data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # load splits
        data = self.add_splits(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
