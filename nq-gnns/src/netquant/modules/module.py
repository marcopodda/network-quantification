from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn

from netquant.modules.utils import load_loss, load_metrics


class NodeClassificationModule(LightningModule):
    def __init__(
        self,
        fold_index: int,
        network: DictConfig,
        criterion: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        metrics: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.fold_index = fold_index
        self.network = hydra.utils.instantiate(network)
        self.classifier = nn.Linear(self.network.hidden_channels, 1)
        self.criterion = load_loss(criterion)
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.logging_params = logging

        main_metric, valid_metric_best, add_metrics = load_metrics(metrics)
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(postfix="/train")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(postfix="/valid")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix="/test")

        self.save_hyperparameters(logger=False)
        self.training_time = None

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, mode="train")
        self.log(
            "loss/train",
            loss,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.train_metric(preds.round().long(), targets)
        self.log(
            "acc/train",
            self.train_metric,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.train_add_metrics(preds, targets)
        self.log_dict(self.train_add_metrics, **self.logging_params)

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, mode="val")
        self.log(
            "loss/valid",
            loss,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.valid_metric(preds.round().long(), targets)
        self.log(
            "acc/valid",
            self.valid_metric,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.valid_add_metrics(preds, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_params)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            "acc/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, mode="test")
        self.log(
            "loss/test",
            loss,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.test_metric(preds.round().long(), targets)
        self.log(
            "acc/test",
            self.test_metric,
            batch_size=preds.size(0),
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        return {"loss": loss}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Any:
        x = self.network(x, edge_index)
        return x

    def predict_embeddings(self, x: torch.Tensor):
        out = self.classifier(x)
        return torch.sigmoid(out).detach()

    def configure_optimizers(self) -> Any:
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_params,
            params=self.parameters(),
            _convert_="partial",
        )
        if not self.slr_params.get("scheduler"):
            return {"optimizer": optimizer}

        scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
            self.slr_params.scheduler,
            optimizer=optimizer,
            _convert_="partial",
        )
        lr_scheduler_dict = {"scheduler": scheduler}
        if self.slr_params.get("extras"):
            for key, value in self.slr_params.get("extras").items():
                lr_scheduler_dict[key] = value
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    def model_step(self, batch: Any, mode: str) -> Any:
        mask = batch[f"{mode}_mask"][self.fold_index][batch.n_id]
        x = self.network(batch.x, batch.edge_index)
        logits = self.classifier(x).view(-1)
        loss = self.criterion(logits[mask], batch.y[mask])
        preds = torch.sigmoid(logits[mask]).detach()
        targets = batch.y[mask].long().detach()
        return loss, preds, targets
