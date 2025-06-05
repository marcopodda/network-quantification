import math
from typing import Any

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from nq.utils.misc import seed_everything


def from_idx_to_mask(indices: list[int], mask_size: int) -> torch.Tensor:
    mask = torch.zeros(mask_size, dtype=torch.bool)
    mask[indices] = 1
    return mask


def create_train_val_test_masks_random(
    graph: Any,
    random_seed: int = 42,
    val_size: float = 0.2,
    calibration_size: float = 0.1,
    num_folds: int = 5,
) -> dict[str, torch.Tensor]:
    seed_everything(random_seed)

    training_masks = torch.empty((num_folds, graph.num_nodes), dtype=torch.bool)
    calibration_masks = torch.empty((num_folds, graph.num_nodes), dtype=torch.bool)
    validation_masks = torch.empty((num_folds, graph.num_nodes), dtype=torch.bool)
    test_masks = torch.empty((num_folds, graph.num_nodes), dtype=torch.bool)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (dev_idx, test_idx) in enumerate(skf.split(torch.arange(graph.num_nodes), graph.y)):
        # split dev in training and validation
        y_dev = graph.y[dev_idx]
        train_idx, val_idx = train_test_split(dev_idx, test_size=math.floor(graph.num_nodes * val_size), random_state=random_seed, stratify=y_dev)

        # split train in training and calibration
        y_train = graph.y[train_idx]
        train_idx, calibration_idx = train_test_split(
            train_idx,
            test_size=math.floor(graph.num_nodes * calibration_size),
            random_state=random_seed,
            stratify=y_train,
        )

        train_perc = len(train_idx) / graph.num_nodes
        cal_perc = len(calibration_idx) / graph.num_nodes
        val_perc = len(val_idx) / graph.num_nodes
        test_perc = len(test_idx) / graph.num_nodes

        print(f"Generated fold {fold + 1} with ratio: {train_perc:.2f} - {cal_perc:.2f} - {val_perc:.2f} - {test_perc:.2f}")

        train_mask = from_idx_to_mask(train_idx, graph.num_nodes)
        calibration_mask = from_idx_to_mask(calibration_idx, graph.num_nodes)
        val_mask = from_idx_to_mask(val_idx, graph.num_nodes)
        test_mask = from_idx_to_mask(test_idx, graph.num_nodes)

        training_masks[fold, :] = train_mask
        calibration_masks[fold, :] = calibration_mask
        validation_masks[fold, :] = val_mask
        test_masks[fold, :] = test_mask

    return {"training_masks": training_masks, "calibration_masks": calibration_masks, "validation_masks": validation_masks, "test_masks": test_masks}
