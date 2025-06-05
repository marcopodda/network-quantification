# Network Quantification with Graph Learning methods

Adapted from: [this](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template) and [this](https://github.com/ashleve/lightning-hydra-template)


## Setup

### 1.Install virtual environment

Requires the `conda` package manager (follow install instructions [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html))

```sh
$ conda create -n netquant python=3.11 -y && conda activate netquant
```

### 2. Install `pytorch`

Tested for PyTorch 2.1.2 with CUDA 12.1. Instructions can be found [here](https://pytorch.org/get-started/locally/).

### 3. Install `torch-geometric` and related libraries

Instructions can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Be sure to install the optional dependencies as well.

### 4. Install package in editable mode

```sh
$ pip install -e .
```

### 5. Export `PROJECT_ROOT`

Run:

```sh
$ echo $PROJECT_ROOT
```

and check whether `PROJECT_ROOT` points to the main directory (where this README file is). If not then run:

```sh
$ export PROJECT_ROOT=$(pwd)
```

Feel free to add the export to your `*sh.rc` so that you don't have to manually do it every time you work on a new shell.


### 6. Train a single configuration

Run the following:

```sh
$ netquant-train experiment=[1] model/network=[2] dataset.fold_index=[3] model.network.num_layers=[4] model.network.hidden_channels=[5]
```

Where:

- `[1]` is the name of the experiment. You can choose among `cora-0-vs-all`, `cora-1-vs-all`, `cora-2-vs-all`, `cora-3-vs-all`, `cora-4-vs-all`, `cora-5-vs-all`, `cora-6-vs-all`, `genius`, `questions`, `tolokers`, and `twitch`.

- `[2]` is the name of the graph learning model. You can choose among `gcn`, `gat`, and `gin`.

- `[3]` is the fold index. Use an integer between `0` and `4`.

- `[4]` is the number of layers of the graph network. In our experiments, we chose among `2`, `3`, and `4`.

- `[5]` is the node embedding dimension. In our experiments, we chose among `128` and `256`.


### 7. Run a quantification evaluation

Run the following:

```sh
$ netquant-quant dataset_name=[1] model_name=[2] fold_index=[3] num_layers=[4] hidden_dim=[5]  task_name=[6]
```

Where arguments `[1]` to `[5]` are the same as above, and:

- `[6]` is the type of evaluation. You can choose among `validation` and `test`.
