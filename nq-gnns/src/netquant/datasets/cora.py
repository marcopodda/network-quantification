from pathlib import Path
from typing import Callable

from torch_geometric.data import Data
from torch_geometric.io import read_planetoid_data

import nqmulti
from .base import BaseDataset


class Cora(BaseDataset):
    """Loads Cora Dataset with custom splits."""

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(
        self,
        root: str = nqmulti.DATA_DIR.as_posix(),
        name: str = "cora-0-vs-all",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name.split("-")[0]
        self.target_class = int(name.split("-")[1])
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name}.{name}" for name in names]

    @property
    def splits_dir(self) -> Path:
        return Path(self.root) / "splits" / f"{self.name}-{self.target_class}-vs-all"

    def get(self, index: int) -> Data:
        data = super().get(index)
        data.y = (data.y == self.target_class).float()
        data = self.add_splits(data)
        return data

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
