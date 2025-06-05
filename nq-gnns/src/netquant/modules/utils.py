import hydra
import torch
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection


def load_metrics(
    metrics_cfg: DictConfig,
) -> tuple[Metric, Metric, MetricCollection]:
    """Load main metric, `best` metric tracker, MetricCollection of additional
    metrics.

    Args:
        metrics_cfg (DictConfig): Metrics config.

    Returns:
        tuple[Metric, Metric, ModuleList]: Main metric, `best` metric tracker,
            MetricCollection of additional metrics.
    """

    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric. Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    additional_metrics = {}
    if metrics_cfg.get("additional"):
        for metric_name, metric_cfg in metrics_cfg.additional.items():
            # print(hydra.utils.instantiate(metric_cfg))
            additional_metrics[metric_name] = hydra.utils.instantiate(metric_cfg)

    return main_metric, valid_metric_best, MetricCollection(additional_metrics)


def load_loss(loss_cfg: DictConfig) -> torch.nn.Module:
    """Load loss module.

    Args:
        loss_cfg (DictConfig): Loss config.

    Returns:
        torch.nn.Module: Loss module.
    """

    weight_params = {}
    for param_name, param_value in loss_cfg.items():
        if "weight" in param_name:
            weight_params[param_name] = torch.tensor(param_value).float()

    loss = hydra.utils.instantiate(loss_cfg, **weight_params)

    return loss
