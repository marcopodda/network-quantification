from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader, NeighborLoader


class DataModule(LightningDataModule):
    """Example of LightningDataModule for single dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return predict dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        fold_index: int,
        dataset: DictConfig,
        loaders: DictConfig,
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            datasets (DictConfig): Datasets config.
            loaders (DictConfig): Loaders config.
            transforms (DictConfig): Transforms config.
        """

        super().__init__()
        self.fold_index = fold_index
        self.cfg_dataset = dataset
        self.cfg_loaders = loaders
        self.dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.dataset:
            self.dataset = hydra.utils.instantiate(self.cfg_dataset)

    def _get_loader_cfg(self, mode: str) -> dict[str, Any]:
        loader_cfg = OmegaConf.to_container(self.cfg_loaders.get(mode))
        if loader_cfg["batch_size"] is None:
            loader_cfg["batch_size"] = self.dataset[0].x.size(0)
        num_layers = loader_cfg.pop("num_layers")
        loader_cfg["num_neighbors"] *= num_layers
        return loader_cfg

    def train_dataloader(self) -> DataLoader | list[DataLoader] | dict[str, DataLoader]:
        data = self.dataset[0]
        input_nodes = torch.arange(data.x.size(0))[data.train_mask[self.fold_index]]
        loader_cfg = self._get_loader_cfg("train")
        return NeighborLoader(data, input_nodes=input_nodes, **loader_cfg)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        data = self.dataset[0]
        input_nodes = torch.arange(data.x.size(0))[data.val_mask[self.fold_index]]
        loader_cfg = self._get_loader_cfg("val")
        return NeighborLoader(data, input_nodes=input_nodes, **loader_cfg)

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        data = self.dataset[0]
        input_nodes = torch.arange(data.x.size(0))[data.test_mask[self.fold_index]]
        loader_cfg = self._get_loader_cfg("test")
        return NeighborLoader(data, input_nodes=input_nodes, **loader_cfg)

    def teardown(self, stage: str | None = None):
        """Clean up after fit or test."""
        pass
