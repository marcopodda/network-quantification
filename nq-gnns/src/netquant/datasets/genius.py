from typing import Callable

import torch
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

import netquant
from netquant.datasets.base import BaseDataset


class Genius(BaseDataset):
    """Load Genius dataset with custom splits."""

    url = "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data"

    def __init__(
        self,
        root: str = netquant.DATA_ROOT.as_posix(),
        name: str = "genius",
        transform: Callable | None = NormalizeFeatures(),
        pre_transform: Callable | None = None,
    ):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["genius.mat"]

    def process(self) -> None:
        mat = loadmat(self.raw_paths[0])

        # node features
        x = torch.from_numpy(mat["node_feat"]).float()

        # node labels
        y = torch.from_numpy(mat["label"]).long()
        y = torch.where(y < 0, 0, y).squeeze().float()

        # adjacency matrix
        edge_index = torch.from_numpy(mat["edge_index"]).long().contiguous()

        # create data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # load splits
        data = self.add_splits(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
