from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class BaseDataset(InMemoryDataset):
    @property
    def raw_dir(self) -> Path:
        return Path(self.root) / self.name / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.root) / self.name / "processed"

    @property
    def splits_dir(self) -> Path:
        return Path(self.root) / "splits" / self.name

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f"{self.url}/{name}", self.raw_dir)

    def add_splits(self, data: Data) -> Data:
        data.train_mask = torch.load(self.splits_dir / "training_masks.pt")
        data.val_mask = torch.load(self.splits_dir / "validation_masks.pt")
        data.cal_mask = torch.load(self.splits_dir / "calibration_masks.pt")
        data.test_mask = torch.load(self.splits_dir / "test_masks.pt")
        return data
