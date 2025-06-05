from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Data

import netquant
from netquant.datasets.base import BaseDataset


class Twitch(BaseDataset):
    """Load Genius dataset with custom splits."""

    url = "https://graphmining.ai/datasets/ptg/twitch"

    def __init__(
        self,
        root: str = netquant.DATA_ROOT.as_posix(),
        name: str = "twitch",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["DE.npz"]

    def process(self) -> None:
        data = np.load(self.raw_paths[0], "r", allow_pickle=True)

        # node features
        x = torch.from_numpy(data["features"]).to(torch.float)

        # node labels
        y = torch.from_numpy(data["target"]).to(torch.long)
        y = torch.where(y < 0, 0, y).squeeze().float()

        # adjacency matrix
        edge_index = torch.from_numpy(data["edges"]).to(torch.long)
        edge_index = edge_index.t().contiguous()

        # create data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # load splits
        data = self.add_splits(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
