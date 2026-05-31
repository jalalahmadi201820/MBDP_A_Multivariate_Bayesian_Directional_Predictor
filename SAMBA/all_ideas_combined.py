

import os
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis, ttest_ind

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)

# ==============================================================================
# GLOBAL NUMERIC SAFETY
# ==============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def sanitize_tensor(x, nan=0.0, posinf=1e4, neginf=-1e4):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def sanitize_numpy(x, nan=0.0, posinf=1e4, neginf=-1e4):
    return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ==============================================================================
# LOSS
# ==============================================================================

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        preds = sanitize_tensor(preds)
        labels = sanitize_tensor(labels)

        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)

            preds = sanitize_tensor(preds)
            labels = sanitize_tensor(labels)

        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        mae = sanitize_tensor(mae)
        return mae
    return loss


# ==============================================================================
# IDEA 1: Bayesian MC Dropout
# ==============================================================================

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def bayesian_inference(model, loader, device, mc_samples=30):
    model.eval()
    enable_dropout(model)

    all_mean_preds = []
    all_targets = []
    all_uncertainty = []

    print(f"\n🔬 Running Bayesian Inference with {mc_samples} MC samples...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Bayesian Sampling"):
            if len(batch) == 2:
                batch_x, batch_y = batch
                batch_stat = None
            elif len(batch) == 3:
                batch_x, batch_y, batch_stat = batch
                batch_stat = sanitize_tensor(batch_stat.to(device))
            else:
                raise ValueError("Unexpected batch format.")

            batch_x = sanitize_tensor(batch_x.to(device))
            batch_y = sanitize_tensor(batch_y.to(device))

            batch_mc_preds = []

            for _ in range(mc_samples):
                if batch_stat is not None:
                    output = model(batch_x, batch_stat)
                else:
                    output = model(batch_x)

                output = sanitize_tensor(output)
                batch_mc_preds.append(output.detach().cpu())

            batch_mc_preds = torch.stack(batch_mc_preds, dim=0)
            batch_mean = batch_mc_preds.mean(dim=0)
            batch_std = batch_mc_preds.std(dim=0)

            batch_mean = sanitize_tensor(batch_mean)
            batch_std = sanitize_tensor(batch_std)

            all_mean_preds.append(batch_mean)
            all_targets.append(batch_y.detach().cpu())
            all_uncertainty.append(batch_std)

    final_preds = torch.cat(all_mean_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    final_uncertainty = torch.cat(all_uncertainty, dim=0)

    return final_preds, final_targets, final_uncertainty


# ==============================================================================
# IDEA 2: T-test Feature Selection
# ==============================================================================

def apply_ttest_feature_selection(csv_path, val_ratio, test_ratio, alpha=0.05, target_col=None):
    print(f"\n🧪 Starting T-test feature selection (alpha={alpha})...")
    df = pd.read_csv(csv_path)

    # sanitize numeric columns early
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    train_ratio = 1.0 - val_ratio - test_ratio
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()

    if target_col is None:
        target_col = df.columns[-1]

    print(f"🎯 Target column: {target_col}")

    if len(train_df[target_col].unique()) > 2:
        print("⚠️ Target is continuous. Binarizing by median for T-test only...")
        median_val = train_df[target_col].median()
        train_labels = (train_df[target_col] > median_val).astype(int)
    else:
        train_labels = train_df[target_col]

    group1_idx = train_labels == 1
    group0_idx = train_labels == 0

    selected_features = []
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == target_col:
            continue

        g1_data = train_df.loc[group1_idx, col]
        g0_data = train_df.loc[group0_idx, col]

        try:
            _, p_val = ttest_ind(g1_data, g0_data, nan_policy='omit', equal_var=False)
            if np.isfinite(p_val) and p_val < alpha:
                selected_features.append(col)
        except Exception:
            continue

    print(f"✅ T-test selected {len(selected_features)} features out of {len(numeric_cols) - 1}")

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    cols_to_keep = non_numeric_cols + selected_features + [target_col]
    cols_to_keep = list(dict.fromkeys(cols_to_keep))

    filtered_df = df[cols_to_keep]
    filtered_csv_path = csv_path.replace('.csv', '_ttest_filtered.csv')
    filtered_df.to_csv(filtered_csv_path, index=False)

    print(f"💾 Filtered dataset saved to: {filtered_csv_path}")
    return filtered_csv_path


# ==============================================================================
# IDEA 3: Statistical Features
# ==============================================================================

def compute_normalized_stats(df_numeric, window=10, train_ratio=0.7):
   
    data = df_numeric.values.astype(np.float32)
    T, N = data.shape
    train_size = int(train_ratio * T)

    train_data = data[:train_size]

    col_min = train_data.min(axis=0)
    col_max = train_data.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range < 1e-8] = 1.0

    data_norm = (data - col_min) / col_range
    data_norm = sanitize_numpy(data_norm)

    stats_mean = np.zeros((T, N), dtype=np.float32)
    stats_std = np.zeros((T, N), dtype=np.float32)
    stats_skew = np.zeros((T, N), dtype=np.float32)
    stats_kurt = np.zeros((T, N), dtype=np.float32)

      for n in range(N):
        series = pd.Series(data_norm[:, n])

        roll = series[:train_size].rolling(window=window, min_periods=max(2, min(window, 3)))
        m = roll.mean()
        s = roll.std()

        m = m.ffill().fillna(float(data_norm[:train_size, n].mean()))
        s = s.ffill().fillna(0.0)

        stats_mean[:train_size, n] = m.values
        stats_std[:train_size, n] = np.maximum(s.values, 0.0)

           stats_mean[train_size:, n] = stats_mean[train_size - 1, n]
        stats_std[train_size:, n] = stats_std[train_size - 1, n]

          stats_skew[train_size:, n] = stats_skew[train_size - 1, n]
        stats_kurt[train_size:, n] = stats_kurt[train_size - 1, n]

    return stats_mean, stats_std, stats_skew, stats_kurt


class StatAugmentedDataset(Dataset):
    def __init__(self, base_dataset, stats_mean, stats_std, stats_skew, stats_kurt, window):
        self.base = base_dataset
        self.stats = np.stack([stats_mean, stats_std, stats_skew, stats_kurt], axis=-1)  # (T, N, 4)
        self.stats = sanitize_numpy(self.stats)
        self.window = window

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.float()

        x = sanitize_tensor(x)
        y = sanitize_tensor(y)

        t_start = idx
        t_end = idx + self.window
        stat_window = self.stats[t_start:t_end]

        if stat_window.shape[0] < self.window:
            pad_len = self.window - stat_window.shape[0]
            if stat_window.shape[0] == 0:
                stat_window = np.zeros((self.window, self.stats.shape[1], 4), dtype=np.float32)
            else:
                pad = np.repeat(stat_window[-1:, :, :], pad_len, axis=0)
                stat_window = np.concatenate([stat_window, pad], axis=0)

        stat_window = np.clip(stat_window, -3.0, 3.0) / 3.0
        stat_window = sanitize_numpy(stat_window)
        stat_window = torch.tensor(stat_window, dtype=torch.float32)

        if x.dim() == 2:
            if x.shape[0] != self.window and x.shape[1] == self.window:
                stat_window = stat_window.permute(1, 0, 2)  # (N, W, 4)
            elif x.shape[0] == self.window:
                pass
            else:
                if x.shape[1] == self.window:
                    stat_window = stat_window.permute(1, 0, 2)

        stat_window = sanitize_tensor(stat_window)
        return x, y, stat_window


class SAMBAWithStatInjection(nn.Module):
    def __init__(self, samba_model, stat_dim=4):
        super().__init__()
        self.samba = samba_model

        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.stat_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, stat_features=None):
        x = sanitize_tensor(x)

        if stat_features is not None:
            stat_features = sanitize_tensor(stat_features)
            stat_proj = self.stat_proj(stat_features).squeeze(-1)
            stat_proj = sanitize_tensor(stat_proj)

            if stat_proj.shape != x.shape:
                raise RuntimeError(
                    f"Shape mismatch after stat projection: x.shape={x.shape}, stat_proj.shape={stat_proj.shape}"
                )

            gate = torch.sigmoid(self.stat_gate)
            x = x + gate * stat_proj
            x = sanitize_tensor(x)

        out = self.samba(x)
        out = sanitize_tensor(out)
        return out


# ==============================================================================
# Custom Trainer
# ==============================================================================

class CombinedTrainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader, args, lr_scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.get('epochs', 100)
        self.patience = args.get('early_stop_patience', 20)
        self.grad_clip = float(args.get('grad_clip', 1.0))

    def _forward_batch(self, batch):
        if len(batch) == 2:
            x, y = batch
            stat = None
        elif len(batch) == 3:
            x, y, stat = batch
            stat = sanitize_tensor(stat.to(self.device))
        else:
            raise ValueError("Unexpected batch structure.")

        x = sanitize_tensor(x.to(self.device))
        y = sanitize_tensor(y.to(self.device))

        if stat is not None:
            pred = self.model(x, stat)
        else:
            pred = self.model(x)

        pred = sanitize_tensor(pred)
        return pred, y

    def train_epoch(self):
        self.model.train()
        losses = []
        skipped = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            pred, y = self._forward_batch(batch)

            if (not torch.isfinite(pred).all()) or (not torch.isfinite(y).all()):
                skipped += 1
                continue

            loss = self.loss_fn(pred, y)
            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            losses.append(float(loss.item()))

        if skipped > 0:
            print(f"⚠️ train: skipped {skipped} non-finite batches")

        return float(np.mean(losses)) if losses else np.inf

    def validate_epoch(self):
        self.model.eval()
        losses = []
        skipped = 0

        with torch.no_grad():
            for batch in self.val_loader:
                pred, y = self._forward_batch(batch)

                if (not torch.isfinite(pred).all()) or (not torch.isfinite(y).all()):
                    skipped += 1
                    continue

                loss = self.loss_fn(pred, y)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                losses.append(float(loss.item()))

        if skipped > 0:
            print(f"⚠️ val: skipped {skipped} non-finite batches")

        return float(np.mean(losses)) if losses else np.inf

    def train(self):
        print("\n🚀 Starting training...")
        best_val = np.inf
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"Epoch [{epoch:03d}/{self.epochs:03d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if np.isfinite(val_loss) and (val_loss < best_val):
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"⏹ Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.test(self.test_loader)

    def test(self, loader):
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                pred, y = self._forward_batch(batch)
                if torch.isfinite(pred).all() and torch.isfinite(y).all():
                    preds.append(pred.detach().cpu())
                    trues.append(y.detach().cpu())

        if len(preds) == 0:
            raise RuntimeError("No finite predictions in test set.")

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        return preds, trues


def build_augmented_loaders(train_loader, val_loader, test_loader, stats_data, window):
    s_mean, s_std, s_skew, s_kurt = stats_data

    train_dataset = StatAugmentedDataset(train_loader.dataset, s_mean, s_std, s_skew, s_kurt, window)
    val_dataset = StatAugmentedDataset(val_loader.dataset, s_mean, s_std, s_skew, s_kurt, window)
    test_dataset = StatAugmentedDataset(test_loader.dataset, s_mean, s_std, s_skew, s_kurt, window)

    batch_size_train = getattr(train_loader, 'batch_size', 32) or 32
    batch_size_val = getattr(val_loader, 'batch_size', 32) or 32
    batch_size_test = getattr(test_loader, 'batch_size', 32) or 32

    train_aug_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=False)
    val_aug_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, drop_last=False)
    test_aug_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False)

    return train_aug_loader, val_aug_loader, test_aug_loader


def safe_returns(y_tensor):
    y_tensor = sanitize_tensor(y_tensor)
    diff = y_tensor[1:] - y_tensor[:-1]
    returns = diff / (y_tensor[:-1].abs() + 1e-6)
    returns = sanitize_tensor(returns)
    return returns


def safe_init_weights(model):
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            # bias / norm weights safer init
            if "bias" in name:
                nn.init.zeros_(p)
            elif "norm" in name.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Combined SAMBA: T-test + Stats Injection + Bayesian MC Dropout")
    parser.add_argument('--dataset', type=str, default='Dataset/combined_dataframe_DJI.csv')
    parser.add_argument('--stats-window', type=int, default=7)
    parser.add_argument('--ttest-alpha', type=float, default=0.05)
    parser.add_argument('--mc-samples', type=int, default=30)
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--no-ttest', action='store_true')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    args_cmd = parser.parse_args()

    model_args, config = get_paper_config()
    _ = get_dataset_info()

    print("🚀 SAMBA Combined Model")
    print("✅ Idea 1: T-test feature selection")
    print("✅ Idea 2: Statistical feature injection")
    print("✅ Idea 3: Bayesian MC Dropout inference")
    print("=" * 80)

    init_seed(config.seed)

    dataset_file = args_cmd.dataset
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset not found: {dataset_file}")
        return

    final_dataset_file = dataset_file
    if not args_cmd.no_ttest:
        final_dataset_file = apply_ttest_feature_selection(
            csv_path=dataset_file,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            alpha=args_cmd.ttest_alpha
        )
    else:
        print("⏭ Skipping T-test feature selection.")

    print("\n📦 Loading and preparing data...")
    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=final_dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    config.num_nodes = num_features
    args = config.to_dict()
    device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    args['device'] = device
    args['grad_clip'] = args_cmd.grad_clip

    # lower lr if too aggressive
    if args.get('lr_init', 1e-3) > 1e-3:
        print(f"⚠️ lr_init={args.get('lr_init')} is high; capping to 1e-3 for stability.")
        args['lr_init'] = 1e-3

    print(f"📈 Number of features after selection: {num_features}")
    print(f"🖥 Device: {device}")

    use_stats = (not args_cmd.no_stats) and (args_cmd.stats_window > 0)
    if use_stats:
        print(f"\n📊 Computing statistical features (window={args_cmd.stats_window})...")
        df_raw = pd.read_csv(final_dataset_file)
        df_numeric = df_raw.select_dtypes(include=[np.number]).copy()
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        train_ratio = 1.0 - config.val_ratio - config.test_ratio
        stats_data = compute_normalized_stats(
            df_numeric=df_numeric,
            window=args_cmd.stats_window,
            train_ratio=train_ratio
        )

        train_loader, val_loader, test_loader = build_augmented_loaders(
            train_loader, val_loader, test_loader, stats_data, config.lag
        )
        print("✅ Statistical augmentation applied.")
    else:
        stats_data = None
        print("⏭ Skipping statistical feature injection.")

    print("\n🧠 Initializing SAMBA model...")
    model_args.vocab_size = num_features

    base_model = SAMBA(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k")
    )

    if use_stats:
        model = SAMBAWithStatInjection(base_model, stat_dim=4)
        print("✅ SAMBA wrapped with statistical injection.")
    else:
        model = base_model

    model = model.to(device)
    safe_init_weights(model)
    print_model_parameters(model, only_num=False)

    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(mmn, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = nn.L1Loss().to(device)
    elif args.get('loss_func') == 'mse':
        loss = nn.MSELoss().to(device)
    else:
        raise ValueError(f"Unknown loss function: {args.get('loss_func')}")

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.get('lr_init', 1e-3),
        eps=1.0e-8,
        weight_decay=1e-5
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
            gamma=0.3
        )

    trainer = CombinedTrainer(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args,
        lr_scheduler=lr_scheduler
    )

    y_pred, y_true = trainer.train()

    print("\n" + "=" * 60)
    print("🚀 Starting Bayesian Inference (Monte Carlo Dropout)")
    print("=" * 60)

    y_pred_bayes, y_true_bayes, y_uncertainty = bayesian_inference(
        trainer.model,
        test_loader,
        device=device,
        mc_samples=args_cmd.mc_samples
    )

    y_p = y_pred_bayes[:, 0, :].numpy()
    y_t = y_true_bayes[:, 0, :].numpy()

    y_p = sanitize_numpy(y_p)
    y_t = sanitize_numpy(y_t)

    y_p = mmn.inverse_transform(y_p)
    y_t = mmn.inverse_transform(y_t)

    y_p = sanitize_numpy(y_p)
    y_t = sanitize_numpy(y_t)

    y_p_torch = torch.tensor(y_p, dtype=torch.float32)
    y_t_torch = torch.tensor(y_t, dtype=torch.float32)

    return_p = safe_returns(y_p_torch)
    return_t = safe_returns(y_t_torch)

    mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
    IC = pearson_correlation(return_t, return_p)
    RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])

    avg_uncertainty = float(sanitize_tensor(y_uncertainty).mean().item())

    print("\n📌 Final Results (Combined Model):")
    print(f"MAE            : {mae:.6f}")
    print(f"RMSE           : {rmse:.6f}")
    print(f"IC             : {IC:.6f}")
    print(f"RIC            : {RIC:.6f}")
    print(f"Avg Uncertainty: {avg_uncertainty:.6f}")

    result_dir = os.path.join("SAMBA_Model", "results")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, "samba_combined_results.txt")
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("Type: Combined (T-test + Statistical Injection + Bayesian MC Dropout)\n")
        f.write(f"Dataset: {final_dataset_file}\n")
        f.write(f"T-test alpha: {args_cmd.ttest_alpha}\n")
        f.write(f"Stats window: {args_cmd.stats_window}\n")
        f.write(f"MC samples: {args_cmd.mc_samples}\n")
        f.write(f"Features: {num_features}\n")
        f.write(f"IC: {IC}\n")
        f.write(f"RIC: {RIC}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"Avg Uncertainty: {avg_uncertainty}\n")
        f.write("-" * 60 + "\n")

    print(f"\n✅ Results saved to: {result_file}")
    print("🎉 Training & Bayesian evaluation completed successfully!")


if __name__ == "__main__":
    main()
