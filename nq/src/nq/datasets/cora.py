from pathlib import Path
from typing import Callable

from torch_geometric.data import Data
from torch_geometric.io import read_planetoid_data

from nq import settings
from nq.datasets.base import BaseDataset


class CoraBinary(BaseDataset):
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(
        self,
        root: str = settings.DATA_DIR.as_posix(),
        name: str = "cora-binary",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name
        self.target_class = 2
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.cora.{name}" for name in names]

    @property
    def splits_dir(self) -> Path:
        return Path(self.root) / "splits" / "cora-binary"

    @property
    def num_classes(self):
        return 2

    def get(self, index: int) -> Data:
        data = super().get(index)
        data.y = (data.y == self.target_class).float()
        data = self.add_splits(data)
        return data

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, "cora")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class Cora(BaseDataset):
    """Loads Cora Dataset with custom splits."""

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(
        self,
        root: str = settings.DATA_DIR.as_posix(),
        name: str = "cora",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name}.{name}" for name in names]

    @property
    def splits_dir(self) -> Path:
        return Path(self.root) / "splits" / f"{self.name}"

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name)

        # load splits
        data = self.add_splits(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
