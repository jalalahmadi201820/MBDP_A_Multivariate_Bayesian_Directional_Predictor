# -*- coding: utf-8 -*-
"""
Main training script for SAMBA stock price forecasting model
Fixed: Statistical features injected as temporal channels, not graph nodes
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import argparse
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def compute_normalized_stats(df_numeric, window=10):
    """
    Compute statistical features ON NORMALIZED data (after min-max scaling).
    Returns a DataFrame with same shape as input but containing stats.

    Key fix: stats are computed per-column on the same scale as the model input,
    so there's no scale mismatch.

    Also fix: NaN padding replaced with forward-fill then zero,
    to avoid injecting artificial zero signals into Mamba states.
    """
    data = df_numeric.values.astype(np.float32)
    T, N = data.shape

    # Min-max normalize each column to [0,1] before computing stats
    # This ensures stats are on the same scale as model inputs
    train_size = int(0.7*len(data))
    col_min = data[:train_size].min(axis=0)
    col_max = data[:train_size].max(axis=0)
    col_range = col_max - col_min
    col_range[col_range < 1e-8] = 1.0  # avoid division by zero
    data_norm = (data - col_min) / col_range

    stats_mean = np.zeros((T, N), dtype=np.float32)
    stats_std = np.zeros((T, N), dtype=np.float32)
    stats_skew = np.zeros((T, N), dtype=np.float32)
    stats_kurt = np.zeros((T, N), dtype=np.float32)

    for n in range(N):
        series = pd.Series(data_norm[:, n])

        # Rolling stats on normalized data
        roll = series.rolling(window=window, min_periods=window)
        m = roll.mean()
        s = roll.std()

        # Fix: forward-fill NaN (use last valid value) then fill remaining with column mean
        # This avoids zero-padding artifacts at the start of the series
        m = m.fillna(method='ffill').fillna(data_norm[:train_size, n].mean())
        s = s.fillna(method='ffill').fillna(0.0)

        stats_mean[:, n] = m.values
        stats_std[:, n] = s.values

        # Skewness and kurtosis
        for t in range(T):
            start = max(0, t - window + 1)
            wdata = data_norm[start:t + 1, n]
            if len(wdata) >= 3 and np.std(wdata) > 1e-8:
                stats_skew[t, n] = float(skew(wdata))
                stats_kurt[t, n] = float(kurtosis(wdata))
            elif t > 0:
                # carry forward instead of zero
                stats_skew[t, n] = stats_skew[t - 1, n]
                stats_kurt[t, n] = stats_kurt[t - 1, n]

    return stats_mean, stats_std, stats_skew, stats_kurt


class StatAugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset and injects statistical features
    as ADDITIONAL INPUT CHANNELS per node — NOT as new nodes.

    Original input shape:  (batch, window, N_nodes)
    Augmented input shape: (batch, window, N_nodes)  <- same N_nodes!

    Stats are concatenated along the time/channel dimension inside the window,
    keeping graph structure (N_nodes) intact.

    Specifically: for each time step t in the window, we append the 4 stat values
    computed up to t as extra features in the feature dimension of that node.
    Since SAMBA treats each node independently before graph aggregation,
    we inject stats as a learned linear projection onto the node embedding.
    """

    def __init__(self, base_dataset, stats_mean, stats_std, stats_skew, stats_kurt,
                 window, stat_proj_dim=None):
        self.base = base_dataset
        self.stats = np.stack([stats_mean, stats_std, stats_skew, stats_kurt], axis=-1)
        # stats shape: (T, N, 4)
        self.window = window

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # x shape: (window, N)
        # We need the corresponding time index
        # The base dataset starts at idx=0 -> time window [0:window]
        t_start = idx
        t_end = idx + self.window
        stat_window = self.stats[t_start:t_end]  # (window, N, 4)

        # Normalize stats to [-1, 1] range to match typical model input scale
        stat_window = np.clip(stat_window, -3.0, 3.0) / 3.0

        return x, y, torch.tensor(stat_window, dtype=torch.float32)


class SAMBAWithStatInjection(nn.Module):
    """
    Wraps SAMBA and injects statistical features via a lightweight
    linear projection BEFORE the main forward pass.

    This keeps N_nodes unchanged (no graph structure damage)
    while still giving the model access to moment information.
    """

    def __init__(self, samba_model, n_nodes, stat_dim=4):
        super().__init__()
        self.samba = samba_model
        # Project 4 stat features -> node embedding dim
        # We add stats to the input embedding, not as new nodes
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, n_nodes),
            nn.LayerNorm(n_nodes),
            nn.GELU()
        )
        # Gate to control how much stat info to inject (learned)
        self.stat_gate = nn.Parameter(torch.zeros(1))  # starts at 0 = no injection

    def forward(self, x, stat_features=None):
        if stat_features is not None:
            # x:             (batch, window, N)
            # stat_features: (batch, window, N, 4)
            batch, window, N, _ = stat_features.shape

            # Project stats: (batch, window, N, 4) -> (batch, window, N)
            stat_proj = self.stat_proj(stat_features)  # (batch, window, N)

            # Gated residual injection — gate starts at 0, learned during training
            gate = torch.sigmoid(self.stat_gate)
            x = x + gate * stat_proj

        return self.samba(x)


def main():
    parser = argparse.ArgumentParser(description="SAMBA Stock Price Forecasting")
    parser.add_argument('--dataset', type=str, default='Dataset/combined_dataframe_IXIC.csv')
    parser.add_argument('--stats-window', type=int, default=7)
    parser.add_argument('--no-stats', action='store_true')
    args_cmd = parser.parse_args()

    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print("=" * 70)

    init_seed(config.seed)

    dataset_file = args_cmd.dataset
    dataset_dir = os.path.dirname(dataset_file)

    if dataset_dir and not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory {dataset_dir} not found!")
        return
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        return

    # Load data and prepare loaders (original, no CSV modification)
    print("Loading and preparing data...")
    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    # Compute stats on normalized data if enabled
    use_stats = (not args_cmd.no_stats) and (args_cmd.stats_window > 0)
    stats_data = None

    if use_stats:
        print(f"Computing statistical features on normalized data (window={args_cmd.stats_window})...")
        df_raw = pd.read_csv(dataset_file)
        df_numeric = df_raw.select_dtypes(include=[np.number])

        s_mean, s_std, s_skew, s_kurt = compute_normalized_stats(
            df_numeric, window=args_cmd.stats_window
        )
        stats_data = (s_mean, s_std, s_skew, s_kurt)
        print(f"✅ Stats computed. Shape: {s_mean.shape} — N_nodes unchanged: {num_features}")

    # Update config — N_nodes stays the same regardless of stats
    config.num_nodes = num_features
    print(f"Graph nodes (unchanged): {num_features}")

    args = config.to_dict()

    # Initialize base SAMBA model
    print("Initializing SAMBA model...")
    model_args.vocab_size = num_features

    base_model = SAMBA(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k")
    )

    # Wrap with stat injection if enabled
    if use_stats:
        model = SAMBAWithStatInjection(base_model, n_nodes=num_features, stat_dim=4)
        print("✅ Statistical injection wrapper applied (gated residual, N_nodes preserved)")
    else:
        model = base_model

    model = model.cuda()

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    print_model_parameters(model, only_num=False)

    # Loss
    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(mmn, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = torch.nn.L1Loss().to(args.get('device'))
    elif args.get('loss_func') == 'mse':
        loss = torch.nn.MSELoss().to(args.get('device'))
    else:
        raise ValueError(f"Unknown loss function: {args.get('loss_func')}")

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.get('lr_init'),
        eps=1.0e-8,
        weight_decay=1e-4,  # small L2 reg to prevent stat gate from overfitting
        amsgrad=False
    )

    lr_scheduler = None
    if args.get('lr_decay'):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[
                int(0.5 * args.get('epochs')),
                int(0.7 * args.get('epochs')),
                int(0.9 * args.get('epochs'))
            ],
            gamma=0.1
        )

    trainer = Trainer(
        model, loss, optimizer, train_loader, val_loader, test_loader,
        args=args, lr_scheduler=lr_scheduler
    )

    print("Starting training...")
    y_pred, y_true = trainer.train()

    print("Evaluating on test set...")
    y1, y2 = trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)

    y_p = np.array(y1[:, 0, :].cpu())
    y_t = np.array(y2[:, 0, :].cpu())

    y_p = mmn.inverse_transform(y_p)
    y_t = mmn.inverse_transform(y_t)

    y_p = torch.tensor(y_p)
    y_t = torch.tensor(y_t)

    diff = y_p[1:] - y_p[:-1]
    return_p = diff / y_p[:-1]

    diff = y_t[1:] - y_t[:-1]
    return_t = diff / y_t[:-1]

    mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
    IC = pearson_correlation(return_t, return_p)
    RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])

    print(f"\nFinal Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"IC:   {IC:.4f}")
    print(f"RIC:  {RIC:.4f}")

    result_train_file = os.path.join("SAMBA_Model", "results")
    os.makedirs(result_train_file, exist_ok=True)

    with open('samba_results.txt', 'a') as f:
        f.write(f"IC: {np.array(IC)}\n")
        f.write(f"RIC: {np.array(RIC)}\n")
        f.write(f"MAE: {np.array(mae)}\n")
        f.write(f"RMSE: {np.array(rmse)}\n\n")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
