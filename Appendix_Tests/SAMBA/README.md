# SAMBA (Graph-Mamba) with Statistical Extensions

A graph-Mamba architecture for stock-price prediction, following *"Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction"*. This folder contains the released baseline and four statistical extensions used in the appendix of the paper.

## Files

| File | Description |
|---|---|
| `SAMBA_default.py` | Released baseline. |
| `SAMBA_with_statistical_features.py` | Adds rolling skewness and excess kurtosis to the node features. |
| `SAMBA_with_t-test.py` | Screens candidate features with a paired *t*-test before training. |
| `SAMBA_with_Baysian.py` | MC-dropout inference with smoothed class priors. |
| `SAMBA_with_BTMF.py` | Combines Graph-Mamba with the Bayesian Temporal Modeling Framework. |
| `paper_config.py` | Model and training configuration matching the reference implementation. |
| `config/` | `ModelArgs` and `TrainingConfig` dataclasses. |
| `models/` | `mamba.py`, `mamba_block.py`, `graph_layers.py`, `normalization.py`, `samba.py`. |
| `trainer/` | Training loop. |
| `utils/` | Data loading, metrics, logging and model utilities. |
| `feature_selector.py` | Feature-screening helpers. |
| `Dataset/` | `combined_dataframe_DJI.csv`, `combined_dataframe_IXIC.csv`, `combined_dataframe_NYSE.csv`. |

## Usage

```bash
pip install -r requirements.txt
python SAMBA_default.py
```

The dataset is selected by the `dataset_file` variable near the top of the script (`Dataset/combined_dataframe_IXIC.csv` by default). Model size, sequence length, state dimension and training schedule come from `paper_config.py`; edit that file to change them.

## Configuration

The statistical hyperparameters are tuned on the last 10% of the training data. Reported metrics are RMSE and information coefficient (IC), expressed as the relative change against the released baseline.
