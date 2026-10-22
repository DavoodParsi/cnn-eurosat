# EuroSAT Land-Use Classification with a Residual CNN

A custom residual CNN for 10-class land-use classification on the EuroSAT RGB dataset, with a focus on **error analysis** and **Grad-CAM interpretability** rather than accuracy alone.

## Results (Phase 1 baseline, validation set)

| Metric | Value |
|---|---|
| Accuracy | 96.23% |
| Macro F1 | 0.9619 |
| Validation loss | 0.1088 |
| Parameters | 3.54M |

The test set is locked and has not been evaluated.

![Training history](docs/figures/training_history.png)
![Confusion matrix](docs/figures/confusion_matrix.png)

### Error analysis

153 of 4,054 validation images are misclassified, and 55 of those errors (36%) have confidence of at least 0.80. The errors cluster around a few visually overlapping class pairs:

- SeaLake → Forest (largest pair, 21 errors)
- PermanentCrop → HerbaceousVegetation
- Pasture → Forest
- AnnualCrop → PermanentCrop
- AnnualCrop → River

Grad-CAM and visual inspection suggest shared vegetation/texture cues and ambiguous blue/green regions. These are treated as hypotheses, not confirmed causes.

![High-confidence error samples](docs/figures/error_samples.png)
![Grad-CAM example](docs/figures/gradcam_example.png)

### Experiment 1: larger input (128×128)

An interrupted run reached 96.69% validation accuracy (loss 0.0973) before training was stopped by hardware limits. Because it did not finish, it is documented as promising but inconclusive. Phase 2 (retraining and hyperparameter search with W&B sweeps) is in progress.

## Model and training

- ResNet-style CNN: stem conv, three residual stages (64/128/256 filters, 2 blocks each), global average pooling, dense head
- Augmentation inside the model: flips, rotation, zoom
- AdamW (learning rate 1e-4, weight decay 1e-4), categorical cross-entropy
- Early stopping, checkpointing and LR reduction on plateau
- Experiment tracking with Weights & Biases

## Repository structure

    notebooks/CNN_High_FLOPs.ipynb   Phase 1 analysis (committed without outputs)
    src/eurosat_cnn/                 Reusable package
        config.py                    Constants and paths
        environment.py               Seeds, GPU and mixed precision
        data.py                      Dataset loading and sanity checks
        models.py                    Augmentation, residual block, CNN
        train.py                     Callbacks, LR schedules, compilation
        evaluate.py                  Metrics
        error_analysis.py            Error and confidence analysis
        explain.py                   Grad-CAM
        visualization.py             Plots and reports
        reporting.py                 Model reports
        tracking.py                  W&B utilities

The notebook is a snapshot of the analysis up to Experiment 1 and contains its own inline code. The `eurosat_cnn` package is the reusable, tested version of the same utilities.

## Setup

Python 3.10 or newer (tested with 3.12).

    pip install -r requirements.txt
    pip install -e .

Place the dataset, already split into train/val/test folders with one subfolder per class, at:

    data/EuroSAT_RGB/train
    data/EuroSAT_RGB/val
    data/EuroSAT_RGB/test

Set the environment variable `EUROSAT_PROJECT_ROOT` to use a different project folder (for example on Google Drive).

Weights & Biases is optional: run `wandb login`, or set `WANDB_MODE=offline`.

## Status

Phase 1 (baseline and error analysis) is complete. Phase 2 is in progress.