# 2D-CNNpred with Statistical Extensions

A 2D convolutional model that predicts the next-day direction of a stock from a tensor of recent price history across several assets. This folder contains the released baseline and four statistical extensions used in the appendix of the paper.

## Files

| File | Description |
|---|---|
| `CNNPred_default.py` | Released baseline. |
| `CNNPred_with_statistical_features.py` | Adds rolling skewness and excess kurtosis to the input features. |
| `CNNPred_with_t-test.py` | Screens candidate features with a paired *t*-test before training. |
| `CNNPred_with_Baysian.py` | MC-dropout inference with smoothed class priors. |
| `CNNPred_with_BTMF.py` | Refines prediction probabilities with market indicators (volatility, returns). |
| `CNNpred2D.py` | Model definition. |
| `processing_data.py`, `dataset.py`, `utils.py` | Data warehouse construction, batching and helpers. |
| `Dataset/` | Processed panels: `Processed_DJI.csv`, `Processed_NASDAQ.csv`, `Processed_NYSE.csv`, `Processed_RUSSELL.csv`, `Processed_SP.csv`. |
| `2D-models/` | Output directory for trained checkpoints and `new results.csv`. |

## Usage

```bash
python CNNPred_default.py --Base-dir .
```

Key options: `--Base-dir` (folder containing `Dataset/`), `--epochs` (default 150), `--seq-len` (default 60), `--predict-day` (default 1), `--num-iter` (default 10 repetitions), `--num-filter`, `--drop`, `--gpu`.

Each variant is run the same way, replacing the script name. Results are written to `2D-models/new results.csv`, and the best checkpoint per repetition is saved as `2D-models/best-<epochs>-<seq_len>-<filters>-<drop>-<run>.pt`.

## Configuration

The statistical hyperparameters are tuned on the last 10% of the training data. Reported metrics are directional accuracy and macro-F1 on the untouched test split.
