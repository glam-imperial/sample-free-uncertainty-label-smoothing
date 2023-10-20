# Propagating Variational Model Uncertainty for Bioacoustic Call Label Smoothing
Project repository for the article by Georgios Rizos, Jenna Lawson, Simon Mitchell, Pranay Shah, Xin Wen, Cristina Banks-Leite, Robert Ewers, Bjoern W. Schuller

https://arxiv.org/abs/2210.10526

## General description

The code is for bioacoustic call detection (binary classification) of 3 second audio clips.

Two datasets are used:
- SAFE-MSMT, which contains annotations for thirty species, mostly birds.
- OSA-SMW, which contains annotations for the whinny call of Geoffrey's spdier monkey.

The preprocessing of the raw acoustic data constitutes normalisation, and storage as TF RECORDS files.

The training/evaluation loop uses the stored TF RECORDS files.

## Datasets

For access to the datasets:
- SAFE-MSMT: please contact Dr. Robert Ewers.
- OSA-SMW: please contact Dr. Jenna L. Lawson or Dr. Cristina Banks-Leite.

## How to use the repo

Follow these steps for replicating the experiments of the article:
- The experiments were performed using Python v3.5.2
- Install all dependencies, summarised in requirements.txt
- Open osa/configuration.py and safe_msmt/configuration.py and edit the PROJECT_FOLDER and DATA_FOLDER variables. The first should point to this code, and the second to the corresponding dataset.
- Execute osa/preprocess.py and safe_msmt/preprocess.py -- these scripts segment the original audio clips into 3 second clips, extract features, perform normalisation, and store the clips in TF RECORDS file format. This may require an hour of processing time.
- Run osa/train.py and safe_msmt/train.py to train the sample-free Bayesian neural network with uncertainty-aware label smoothing.

The training configuration is summarised in corresponding YAML files in the osa/experiment_configurations and safe_msmt/experiment_configurations folders.

If you would like to run the other models from the paper, you can edit train.py such that it points to a different YAML file.
