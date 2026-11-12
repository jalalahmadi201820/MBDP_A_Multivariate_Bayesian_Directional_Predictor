

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm  # For progress bar functionality
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer



def enable_dropout(model):
    """
    Forces Dropout layers to be in training mode even during evaluation.
    This is essential for Monte Carlo Dropout to work.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def bayesian_inference(model, loader, device, mc_samples=30):
    """
    Performs Bayesian Inference using Monte Carlo Dropout.

    Args:
        model: The trained neural network.
        loader: Data loader for the test set.
        device: 'cuda' or 'cpu'.
        mc_samples: Number of stochastic forward passes (Higher = better estimation but slower).

    Returns:
        mean_preds: The averaged prediction (Bayesian Point Estimate).
        std_preds: The standard deviation (Uncertainty).
        targets: The true labels.
    """
    model.eval()
    enable_dropout(model)  # <--- The Magic: Keep randomness alive

    all_mean_preds = []
    all_targets = []

    print(f"🔬 Running Bayesian Inference with {mc_samples} MC samples...")

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Bayesian Sampling"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Store 'mc_samples' predictions for this batch
            batch_mc_preds = []

            for _ in range(mc_samples):
                output = model(batch_x)
                batch_mc_preds.append(output.cpu())  # Move to CPU to save GPU RAM

            # Stack: [Samples, Batch, Nodes, Horizon]
            batch_mc_preds = torch.stack(batch_mc_preds)

            # Calculate Bayesian Statistics
            # Mean captures the "wisdom of crowds" (Ensemble effect)
            batch_mean = batch_mc_preds.mean(dim=0)

            all_mean_preds.append(batch_mean)
            all_targets.append(batch_y.cpu())

    # Concatenate all batches
    final_preds = torch.cat(all_mean_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    return final_preds, final_targets


# ==============================================================================
# 📉 LOSS FUNCTION
# ==============================================================================

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



def main():
    """Main training function using paper configuration"""
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA (Bayesian Enhanced): Stock Price Prediction")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print("=" * 70)

    init_seed(config.seed)

    # 1. Data Preparation
    print("Loading and preparing data...")
    # Take the average across all datasets.
    dataset_file = 'Dataset/combined_dataframe_IXIC.csv'

    # Check Dataset existence (kept from original code logic)
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        return

    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    config.num_nodes = num_features
    args = config.to_dict()
    device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Number of features: {num_features} | Device: {device}")

    # 2. Model Initialization
    print("Initializing SAMBA model...")
    model_args.vocab_size = num_features

    model = SAMBA(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k")
    ).to(device)

    # Parameter Initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    print_model_parameters(model, only_num=False)

    # 3. Setup Loss & Optimizer
    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(mmn, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = torch.nn.L1Loss().to(device)
    elif args.get('loss_func') == 'mse':
        loss = torch.nn.MSELoss().to(device)
    else:
        raise ValueError(f"Unknown loss function")

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.get('lr_init'),
        eps=1.0e-8,
        weight_decay=0
    )

    lr_scheduler = None
    if args.get('lr_decay'):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(0.5 * args['epochs']), int(0.7 * args['epochs']), int(0.9 * args['epochs'])],
            gamma=0.1
        )

    # 4. Training Loop
    trainer = Trainer(
        model, loss, optimizer, train_loader, val_loader, test_loader,
        args=args, lr_scheduler=lr_scheduler
    )

    print("Starting standard training...")
    trainer.train()

    print("\n" + "=" * 50)
    print("🚀 Starting Bayesian Inference (Monte-Carlo Dropout)")
    print("=" * 50)

    # We use the custom bayesian_inference function instead of standard trainer.test
    # mc_samples=30 is a good balance between accuracy and speed
    y_pred_bayesian, y_true_tensor = bayesian_inference(
        trainer.model,
        test_loader,
        device=device,
        mc_samples=50
    )

    # 5. Metrics Calculation & Inversion
    # Move to numpy for processing
    y_p = y_pred_bayesian[:, 0, :].numpy()  # Taking the first horizon step if multi-step
    y_t = y_true_tensor[:, 0, :].numpy()

    # Inverse Transform
    y_p = mmn.inverse_transform(y_p)
    y_t = mmn.inverse_transform(y_t)

    # Back to tensor for metric functions
    y_p_torch = torch.tensor(y_p)
    y_t_torch = torch.tensor(y_t)

    # Calculate Returns (as per original logic)
    # Note: Adding epsilon to avoid division by zero
    diff_p = y_p_torch[1:] - y_p_torch[:-1]
    return_p = diff_p / (y_p_torch[:-1] + 1e-8)

    diff_t = y_t_torch[1:] - y_t_torch[:-1]
    return_t = diff_t / (y_t_torch[:-1] + 1e-8)

    # Compute Metrics
    mae, rmse, _ = All_Metrics(return_p, return_t, None, None)
    IC = pearson_correlation(return_t, return_p)
    RIC = rank_information_coefficient(return_t[:, 0], return_p[:, 0])

    print(f"\n Final Results (Bayesian Enhanced):")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"IC   : {IC:.6f} (Higher is better)")
    print(f"RIC  : {RIC:.6f} (Higher is better)")

    # Save results
    result_train_file = os.path.join("SAMBA_Model", "results")
    os.makedirs(result_train_file, exist_ok=True)

    with open(os.path.join(result_train_file, 'samba_bayesian_results.txt'), 'a') as f:
        f.write(f"Type: Bayesian MC-Dropout (Samples=30)\n")
        f.write(f"IC: {IC}\n")
        f.write(f"RIC: {RIC}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write("-" * 30 + "\n")

    print("\n Bayesian Training & Evaluation completed successfully!")


if __name__ == "__main__":
    main()
