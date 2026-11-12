# DANSMP with Statistical Extensions

A graph-based stock-movement model that combines numerical price features with textual signals through a Graph CNN. This folder contains the released baseline and four statistical extensions used in the appendix of the paper.

## Files

| File | Description |
|---|---|
| `DANSMP_default.py` | Released baseline. |
| `DANSMP_with_statistical_features.py` | Adds rolling skewness and excess kurtosis to the node features. |
| `DANSMP_with_t-test.py` | Screens candidate features with a paired *t*-test before training. |
| `DANSMP_with_Baysian.py` | MC-dropout inference with smoothed class priors. |
| `DANSMP_with_BTMF.py` | Combines the graph model with sentiment features and the Bayesian Temporal Modeling Framework. |
| `Model.py`, `Layers.py` | Graph CNN and layer definitions. |
| `stat.py`, `utils.py` | Statistical descriptors and training helpers. |
| `data/` | Processed CSI100E and CSI300 tensors (`x_num_standard.pkl`, `x_newtext.pkl`, `edge_new.pkl`, `y_1.pkl`, `interactive.pkl`). |
| `SavedModels/` | Output directory for checkpoints. |

## Usage

```bash
python DANSMP_default.py --max-epoch 400 --wait-epoch 30
```

Options: `--max-epoch` (default 400), `--wait-epoch` (early-stopping patience, default 30). The data paths are relative (`./data/...`), so run the script from inside this folder.

Each variant is run the same way, replacing the script name. Training logs AUC and accuracy on both the evaluation and test splits each epoch, and the best evaluation AUC is retained.

## Configuration

The statistical hyperparameters are tuned on the last 10% of the training data. Reported metrics are AUC and accuracy.
