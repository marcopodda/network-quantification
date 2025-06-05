## Setup

1. Create conda environment

```bash
conda create -n nq python=3.12 && conda activate nq
```

2. Install packages
```bash
pip install -e .
```

3. Install `PyG` dependencies (match `cu124` with your `CUDA` version)
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html --force-reinstall --no-cache-dir
```

4. Set up `PROJECT_ROOT` to this directory, `DATA_DIR` and `OUTPUT_DIR` in a .env file in the root directory (otherwise they will be placed in the root directory)


## Training

1. Ensure you are in the correct Python environment.

2. Run:

```bash
nq-train dataset=[DATASET] model=[MODEL] fold_index=[FOLD_INDEX] trial=[TRIAL]
```

Where:
- `DATASET` is any among `amazon`, `cora`, `cora-binary`, `flickr`, `genius`, `questions`, `toloker`, `twitch`
- `MODEL` is any among `lr`, `cdq`, `enq`, `wvrn`, `gesn`
- `FOLD_INDEX` is an interger from `0` to `4`
- `TRIAL` depends on the model.
  - if `MODEL` is `gesn` or `lr`, `TRIAL` is an integer between `0` and `99`.
  - if `MODEL` is `cdq` or `enq`, `TRIAL` is either `0` or `1`.
  - if `MODEL` is `wvrn`, `TRIAL` is an integer between `0` and `4`.

Results will be placed in the `OUTPUT_DIR` you specified in the `.env` file.

## Quantification

1. Ensure you are in the correct Python environment.

2. Run:

```bash
nq-quant dataset=[DATASET] model=[MODEL] fold_index=[FOLD_INDEX] trial=[TRIAL] task_name=[TASK_NAME]
```

Where:
- `DATASET` is any among `amazon`, `cora`, `cora-binary`, `flickr`, `genius`, `questions`, `toloker`, `twitch`
- `MODEL` is any among `lr`, `cdq`, `enq`, `wvrn`, `gesn`
- `FOLD_INDEX` is an interger from `0` to `4`
- `TRIAL` depends on the model.
  - if `MODEL` is `gesn` or `lr`, `TRIAL` is an integer between `0` and `99`.
  - if `MODEL` is `cdq` or `enq`, `TRIAL` is either `0` or `1`.
  - if `MODEL` is `wvrn`, `TRIAL` is an integer between `0` and `4`.
- `TASK_NAME` is either `validation` or `test`.

Results will be placed in the `OUTPUT_DIR` you specified in the `.env` file.
