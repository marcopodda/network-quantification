from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import netquant
from netquant.quantification import DATASETS, DATASETS_CAT, METHODS, METHODS_CAT, MODELS, MODELS_CAT


def get_metric(root: Path, metric: str) -> float:
    metric_name = f"{metric}/test"
    metrics = pd.read_csv(root / "csv" / "version_0" / "metrics.csv")
    metrics = metrics[metrics[metric_name].notnull()]
    return float(metrics.iloc[0][metric_name])


def evaluate_best(dataset_name, model_name, fold_index):
    root = netquant.EXP_ROOT / dataset_name / model_name / "train" / f"fold_{fold_index}"
    val_scores = {}
    for config_dir in root.iterdir():
        qdir = config_dir / "quantification" / "validation"
        with open(qdir / "scores.yaml", "r") as file:
            val_results = yaml.load(file, Loader=yaml.FullLoader)
        del val_results["CC"]
        winner_method = min(val_results, key=lambda k: val_results[k])
        val_scores[config_dir.stem] = val_results[winner_method]

    val_winner = min(val_scores, key=lambda k: val_scores[k])
    winner_dir = root / val_winner
    qdir = winner_dir / "quantification" / "test"
    with open(qdir / "scores.yaml", "r") as file:
        test_results = yaml.load(file, Loader=yaml.FullLoader)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "fold": fold_index,
        "config": val_winner,
        "mae_val": val_scores[val_winner],
        "mae_test": test_results[winner_method],
    }


def evaluate_method(dataset_name, model_name, method, fold_index) -> dict[str, Any]:
    root = netquant.EXP_ROOT / dataset_name / model_name / "train" / f"fold_{fold_index}"

    val_scores = {}
    for config_dir in root.iterdir():
        qdir = config_dir / "quantification" / "validation"
        with open(qdir / "scores.yaml", "r") as file:
            val_results = yaml.load(file, Loader=yaml.FullLoader)
        val_scores[config_dir.stem] = val_results[method]

    val_winner = min(val_scores, key=lambda k: val_scores[k])

    winner_dir = root / val_winner
    qdir = winner_dir / "quantification" / "test"
    with open(qdir / "scores.yaml", "r") as file:
        test_results = yaml.load(file, Loader=yaml.FullLoader)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "fold": fold_index,
        "config": val_winner,
        "mae_val": val_scores[val_winner],
        "mae_test": test_results[method],
    }


def evaluate_all_methods() -> pd.DataFrame:
    rows = []

    for dataset_name in DATASETS:
        for model_name in MODELS:
            for method in METHODS:
                for fold_index in range(5):
                    r = evaluate_method(dataset_name, model_name, method, fold_index)
                    rows.append(r)

    data = pd.DataFrame(rows)
    data.method = data.method.astype(METHODS_CAT)
    data.dataset = data.dataset.astype(DATASETS_CAT)
    data.model = data.model.astype(MODELS_CAT)
    return (
        data.drop("config", axis=1)
        .groupby(["dataset", "model", "method"])
        .agg(["mean", "std"])
        .drop("fold", axis=1)
        .round(3)
    )


def evaluate_best_methods():
    rows = []

    for dataset_name in DATASETS:
        for model_name in MODELS:
            for fold_index in range(5):
                r = evaluate_best(dataset_name, model_name, fold_index)
                rows.append(r)

    data = pd.DataFrame(rows)
    data.dataset = data.dataset.astype(DATASETS_CAT)
    data.model = data.model.astype(MODELS_CAT)
    return data.drop("config", axis=1).groupby(["dataset", "model"]).agg(["mean", "std"]).drop("fold", axis=1).round(3)


def evaluate_cc():
    rows = []
    for dataset_name in DATASETS:
        for model_name in MODELS:
            for fold_index in range(5):
                r = evaluate_method(dataset_name, model_name, "CC", fold_index)
                rows.append(r)

    data = pd.DataFrame(rows)
    data.dataset = data.dataset.astype(DATASETS_CAT)
    data.model = data.model.astype(MODELS_CAT)
    return (
        data.drop("config", axis=1)
        .groupby(["dataset", "model", "method"])
        .agg(["mean", "std"])
        .drop("fold", axis=1)
        .round(3)
    )


def evaluate_timing_dataset(dataset: str) -> list[dict[str, Any]]:
    r = netquant.EXP_ROOT / dataset
    rows = []
    for model in ["gcn", "gat", "gin"]:
        timing = []
        for f in (r / model).glob("**/time.txt"):
            with f.open() as file:
                timing.append(float(file.readlines()[0].rstrip()))
        if timing != []:
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Time": np.mean(timing),
                    "TimeStd": np.std(timing),
                }
            )
    return rows


def evaluate_timing_all() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        rows += evaluate_timing_dataset(dataset)
    return pd.DataFrame(rows).round(1)
