

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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


def apply_ttest_feature_selection(csv_path, val_ratio, test_ratio, alpha=0.05, target_col=None):

    print(f" Starting T-test feature selection (alpha = {alpha})...")
    df = pd.read_csv(csv_path)

    train_ratio = 1.0 - val_ratio - test_ratio
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()

    if target_col is None:
        target_col = df.columns[-1]

    print(f" Target column identified as: {target_col}")

    if len(train_df[target_col].unique()) > 2:
        print(" Target seems continuous! Binarizing it just for the t-test (1 for > median, 0 for <= median)...")
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

       
        t_stat, p_val = ttest_ind(g1_data, g0_data, nan_policy='omit')

        if p_val < alpha:
            selected_features.append(col)

    print(f" T-test done! Selected {len(selected_features)} VIP features out of {len(numeric_cols) - 1}.")

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    cols_to_keep = non_numeric_cols + selected_features + [target_col]

    cols_to_keep = list(dict.fromkeys(cols_to_keep))

    filtered_df = df[cols_to_keep]
    filtered_csv_path = csv_path.replace('.csv', '_ttest_filtered.csv')
    filtered_df.to_csv(filtered_csv_path, index=False)

    print(f"💾 Filtered dataset saved to: {filtered_csv_path}")
    return filtered_csv_path


def main():
    """Main training function using paper configuration"""
    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print(" SAMBA: A Graph-Mamba Approach for Stock Price Prediction")
    print(f" Paper: {dataset_info['paper_title']}")
    print(f"️  Conference: {dataset_info['conference']}")
    print(f" Authors: {', '.join(dataset_info['authors'])}")
    print(f" Expected Features: {dataset_info['total_features']}")
    print("=" * 70)

    # Initialize seed for reproducibility
    init_seed(config.seed)

    # Prepare data
    print("Loading and preparing data...")

    available_datasets = [ds['file'] for ds in dataset_info['datasets']]
    # Take the average across all datasets.

    dataset_file = 'Dataset/combined_dataframe_IXIC.csv'

    if not os.path.exists('Dataset'):
        print(" Dataset folder not found!")
        print("Please create a 'Dataset' folder and put your CSV files in it.")
        return

    if not os.path.exists(dataset_file):
        print(f" Dataset {dataset_file} not found!")
        return

       SIGNIFICANCE_LEVEL = 0.05
    filtered_dataset_file = apply_ttest_feature_selection(
        csv_path=dataset_file,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        alpha=SIGNIFICANCE_LEVEL
    )

    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=filtered_dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    # Update config with actual number of features (nodes in the graph)
    config.num_nodes = num_features
    print(f"📈 Number of final features (graph nodes): {num_features}")

    # Convert config to dict for compatibility
    args = config.to_dict()

    # Initialize model with paper configuration
    print("Initializing SAMBA model...")
    model_args.vocab_size = num_features

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

    print("Training completed successfully! ")


if __name__ == "__main__":
    main()
