
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
    """Masked MAE loss function"""

    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        from utils.metrics import MAE_torch
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def enhance_dataset(csv_file, window=10):
    """
    Enhance the dataset by adding statistical features for each numeric feature/stock column:
    - Rolling mean (window)
    - Rolling std (window)
    - Skewness (window)
    - Kurtosis (window)

    Non-numeric columns (e.g., 'Name') are preserved unchanged.
    Returns path to the enhanced CSV file.
    """
    print(f"Enhancing dataset with statistical features (window={window})...")

    # Load original CSV
    df = pd.read_csv(csv_file)

    # Identify numeric and non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset!")

    data = df_numeric.values  # T x N_numeric
    T, N = data.shape

    print(f"Original numeric shape: {T} time steps x {N} features")

    # Compute stats on numeric data only
    stats = np.zeros((T, 4 * N))
    for n in range(N):
        series = data[:, n]
        pd_series = pd.Series(series)

        # Rolling mean and std (using pandas for efficiency)
        means = pd_series.rolling(window=window).mean().fillna(0).values
        stds = pd_series.rolling(window=window).std().fillna(0).values

        # Skewness and kurtosis (loop required)
        skews = np.zeros(T)
        kurts = np.zeros(T)
        for t in range(T):
            if t >= window - 1:
                wdata = series[t - window + 1: t + 1]
                if len(wdata) == window and np.std(wdata) > 1e-8:
                    skews[t] = skew(wdata)
                    kurts[t] = kurtosis(wdata)

        # Assign to stats
        stats[:, n] = means
        stats[:, N + n] = stds
        stats[:, 2 * N + n] = skews
        stats[:, 3 * N + n] = kurts

    # Enhanced numeric data: original numeric + stats (5x original numeric features)
    enhanced_numeric_data = np.hstack((data, stats))
    new_N_numeric = enhanced_numeric_data.shape[1]

    # Create meaningful column names for enhanced numeric
    new_numeric_cols = []
    stat_types = ['_mean', '_std', '_skew', '_kurt']
    for col in numeric_cols:
        new_numeric_cols.append(col)  # original numeric column
        for stype in stat_types:
            new_numeric_cols.append(f"{col}{stype}")

    enhanced_numeric_df = pd.DataFrame(
        enhanced_numeric_data,
        columns=new_numeric_cols,
        index=df_numeric.index
    )

    # Preserve non-numeric columns (e.g., 'Name') and join with enhanced numeric
    if non_numeric_cols:
        df_non_numeric = df[non_numeric_cols]
        enhanced_df = df_non_numeric.join(enhanced_numeric_df)
        print(f"Preserved {len(non_numeric_cols)} non-numeric columns (e.g., 'Name')")
    else:
        enhanced_df = enhanced_numeric_df

    # Save enhanced file
    enhanced_file = csv_file.replace('.csv', '_enhanced_stats.csv')
    enhanced_df.to_csv(enhanced_file, index=True)  # Save index (e.g., Date)

    print(f"Enhanced shape: {len(enhanced_df)} rows x {len(enhanced_df.columns)} columns")
    print(f"Enhanced numeric features: {new_N_numeric} (5x original {N})")
    print(f"Enhanced file saved: {enhanced_file}")

    return enhanced_file


def main():
    """Main training function using paper configuration"""
    # Command-line arguments
    parser = argparse.ArgumentParser(description="SAMBA Stock Price Forecasting")
    # Take the average across all datasets.

    parser.add_argument('--dataset', type=str, default='Dataset/combined_dataframe_IXIC.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--stats-window', type=int, default=7,
                        help='Window size for statistical features (0 to disable)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Disable statistical feature enhancement')
    args_cmd = parser.parse_args()

    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print(f"🏛️  Conference: {dataset_info['conference']}")
    print(f"👥 Authors: {', '.join(dataset_info['authors'])}")
    print(f"📊 Expected Features: {dataset_info['total_features']}")
    print("=" * 70)

    # Initialize seed for reproducibility
    init_seed(config.seed)

    # Prepare data path
    dataset_file = args_cmd.dataset

    # Check if Dataset folder exists
    dataset_dir = os.path.dirname(dataset_file)
    if dataset_dir and not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory {dataset_dir} not found!")
        return

    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        print("Available CSV files:")
        if dataset_dir:
            for file in os.listdir(dataset_dir):
                if file.endswith('.csv'):
                    print(f"  ✅ {os.path.join(dataset_dir, file)}")
        return

    # Enhance dataset if enabled
    if not args_cmd.no_stats and args_cmd.stats_window > 0:
        dataset_file = enhance_dataset(dataset_file, window=args_cmd.stats_window)

    # Prepare data loaders
    print("Loading and preparing data...")
    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    # Update config with actual number of features (nodes in the graph)
    config.num_nodes = num_features
    print(f"Number of features (graph nodes): {num_features}")

    # Convert config to dict for compatibility
    args = config.to_dict()

    # Initialize model with paper configuration
    print("Initializing SAMBA model...")
    model_args.vocab_size = num_features  # Update with actual number of features

    model = SAMBA(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k")
    )

    model = model.cuda()

    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    print_model_parameters(model, only_num=False)

    # Setup loss function
    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(mmn, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = torch.nn.L1Loss().to(args.get('device'))
    elif args.get('loss_func') == 'mse':
        loss = torch.nn.MSELoss().to(args.get('device'))
    else:
        raise ValueError(f"Unknown loss function: {args.get('loss_func')}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.get('lr_init'),
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )

    # Setup learning rate scheduler
    lr_scheduler = None
    if args.get('lr_decay'):
        print('Applying learning rate decay.')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[0.5 * args.get('epochs'), 0.7 * args.get('epochs'), 0.9 * args.get('epochs')],
            gamma=0.1
        )

    # Initialize trainer
    trainer = Trainer(
        model, loss, optimizer, train_loader, val_loader, test_loader,
        args=args, lr_scheduler=lr_scheduler
    )

    # Start training
    print("Starting training...")
    y_pred, y_true = trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    y1, y2 = trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)

    # Convert predictions and targets
    y_p = np.array(y1[:, 0, :].cpu())
    y_t = np.array(y2[:, 0, :].cpu())

    # Inverse transform to original scale
    y_p = mmn.inverse_transform(y_p)
    y_t = mmn.inverse_transform(y_t)

    # Convert to tensors
    y_p = torch.tensor(y_p)
    y_t = torch.tensor(y_t)

    # Calculate returns
    diff = y_p[1:] - y_p[:-1]
    return_p = diff / y_p[:-1]

    diff = y_t[1:] - y_t[:-1]
    return_t = diff / y_t[:-1]

    # Calculate metrics
    mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
    IC = pearson_correlation(return_t, return_p)
    RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])

    print(f"Final Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Information Coefficient (IC): {IC:.4f}")
    print(f"Rank Information Coefficient (RIC): {RIC:.4f}")

    # Save results
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
