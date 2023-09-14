# PyTorch implementation of nnPUss / nnPUcc algorithm

Implementation repository for "Single-sample versus case-control sampling schemes for Positive Unlabeled data: the story of two scenarios" paper.

## Requirements

The preferred way to install requirements is via `conda` or `mamba`: you can run 
    
    conda env create -f .\env.yml

or 

    mamba env create -f .\env.yml

to create environment containing required packages and

    conda activate nnPU-env

to activate it prior to script execution.

You can also use `pip` to install the requirements; list of dependencies is in the `env.yml` file in the repository root.

## Experiment replication

In order to run experiment suite described in the paper, run 

    python ./src/nnPUss/main.py

## Repository structure

Main implementation of the algorithm can be found in `src/nnPUss` directory. The core of the algorithm (implementation of the nnPUss and nnPUcc risk functions) can be found in `loss.py`. `run_experiment.py` contains a single experiment run code: main training loop and testing, using configurations obtained from `main.py`. The remaining files defined various other components:
- `dataset.py` - data preprocessing and PU dataset generation; note especially `SCAR_SS_Labeler` and `SCAR_SS_Labeler` classes used for labeling,
- `model.py` - neural architectures used for training,
- `experiment_config.py` - single experiment configuration,
- `dataset_configs.py` - dataset specific experiment configurations,
- `metric_values.py` - metrics class.

Some of the files in the directory are used for result generation:
- `read_results.py` - result table generation,
- `dataset_stats.py` - dataset statistics table,
- `extra-plots` directory - all figures present in the paper.

Repository root contain two more important elements:
- `env.yml` - list of dependencies,
- `test/` - tests for labeling scenarios.

## Paper supplement

You can find the paper supplement in the `Single-sample versus case-control sampling schemes for Positive Unlabeled data: the story of two scenarios – supplementary material.pdf` file in the repository root.