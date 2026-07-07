import os
import torch
import torch.nn as nn
import numpy as np
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    prepare_data, init_seed, print_model_parameters,
    pearson_correlation, rank_information_coefficient, All_Metrics
)
from trainer import Trainer


class BTMF(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

        self._alpha_h = nn.Parameter(torch.tensor(self._inv_sigmoid(0.2)))
        self._alpha_c = nn.Parameter(torch.tensor(self._inv_sigmoid(0.2)))
        self._k_up    = nn.Parameter(torch.tensor(1.0))
        self._k_down  = nn.Parameter(torch.tensor(1.0))

        self._prior_decay = nn.Parameter(torch.tensor(self._inv_sigmoid(0.5)))

        self.sharpness = 10.0

    @staticmethod
    def _inv_sigmoid(p):
        return float(np.log(p / (1.0 - p)))

    def _soft_sign(self, x):
        return torch.tanh(self.sharpness * x)

    def _soft_gt(self, x, thr):
        return torch.sigmoid(self.sharpness * (x - thr))

    @staticmethod
    def _compute_features(x):

        eps = 1e-8
        prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        diff = x - prev

        short_ret = diff / (prev.abs() + eps)

        t_idx = torch.arange(1, x.size(1) + 1, device=x.device,
                             dtype=x.dtype).view(1, -1, 1)
        long_ret = torch.cumsum(short_ret, dim=1) / t_idx

        gain = torch.relu(diff)
        loss = torch.relu(-diff)
        avg_gain = torch.cumsum(gain, dim=1) / t_idx
        avg_loss = torch.cumsum(loss, dim=1) / t_idx
        rs = avg_gain / (avg_loss + eps)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        mean_r = long_ret
        var_r = torch.cumsum((short_ret - mean_r) ** 2, dim=1) / t_idx
        volatility = torch.sqrt(var_r + eps)

        return short_ret, long_ret, rsi, volatility

    def forward(self, x):

        B, T, N = x.shape
        alpha_h = torch.sigmoid(self._alpha_h)
        alpha_c = torch.sigmoid(self._alpha_c)
        k_up    = self._k_up
        k_down  = self._k_down
        decay   = torch.sigmoid(self._prior_decay)

        short_ret, long_ret, rsi, vol = self._compute_features(x)

        h1 = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        h2 = torch.zeros_like(h1)
        c1 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)
        prior_up   = torch.full_like(h1, 0.5)
        prior_down = torch.full_like(h1, 0.5)

        for t in range(T):
            sr, lr = short_ret[:, t, :], long_ret[:, t, :]
            rs, vl = rsi[:, t, :], vol[:, t, :]

            h1 = (1.0 - alpha_h) * h1 + alpha_h * sr
            h2 = (1.0 - alpha_h) * h2 + alpha_h * lr
            c1 = (1.0 - alpha_c) * c1 + alpha_c * h1
            c2 = (1.0 - alpha_c) * c2 + alpha_c * h2

            prior_up   = decay * prior_up   + (1.0 - decay) * 0.5
            prior_down = decay * prior_down + (1.0 - decay) * 0.5

            s_sr = self._soft_sign(sr)
            s_lr = self._soft_sign(lr)

            score_up = s_sr + s_lr
            # (rsi<70 ? +0.5 : -0.5)  →  0.5 - 1.0*sigmoid(rsi-70)
            score_up = score_up + (0.5 - 1.0 * self._soft_gt(rs, 80.0))
            score_up = score_up - vl * 0.2

            score_down = -s_sr - s_lr
            # (rsi>30 ? 0.3 : 0.8)  →  0.8 - 0.5*sigmoid(rsi-30)
            score_down = score_down + (0.8 - 0.5 * self._soft_gt(rs, 20.0))
            score_down = score_down + vl * 0.2

            like_up   = torch.nn.functional.softplus(1.0 + k_up   * score_up)   + 0.01
            like_down = torch.nn.functional.softplus(1.0 + k_down * score_down) + 0.01

            post_up   = like_up   * prior_up
            post_down = like_down * prior_down
            norm = post_up + post_down + 1e-8
            prior_up   = post_up / norm
            prior_down = post_down / norm

        states = torch.stack([h1, h2, c1, c2], dim=-1)  # [B, N, 4]
        return prior_up, prior_down, states



class SAMBA_BTMF(nn.Module):
    def __init__(self, model_args, hid, lag, horizon, embed_dim, cheb_k, num_nodes):
        super().__init__()
        self.samba = SAMBA(model_args, hid, lag, horizon, embed_dim, cheb_k)
        self.btmf = BTMF(num_nodes)
        self.horizon = horizon
        self.fusion_gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
  
        out = self.samba(x)

        prob_up, prob_down, states = self.btmf(x)
        bayes_feat = torch.cat(
            [prob_up.unsqueeze(-1), prob_down.unsqueeze(-1), states], dim=-1
        )

        gate = self.fusion_gate(bayes_feat).squeeze(-1)

        confidence = (prob_up - prob_down)

        correction = 1.0 + self.gamma * gate * confidence   # [B, N]
        correction = correction.unsqueeze(1)                 # [B, 1, N]

        return out * correction


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
    """Main training function using paper configuration + BTMF fusion"""
    # Get paper configuration
    model_args, config = get_paper_config()
    dataset_info = get_dataset_info()

    print("🚀 SAMBA + BTMF: Graph-Mamba with Bayesian-LSTM Fusion")
    print(f"📚 Paper: {dataset_info['paper_title']}")
    print(f"🏛️  Conference: {dataset_info['conference']}")
    print(f"👥 Authors: {', '.join(dataset_info['authors'])}")
    print(f"📊 Expected Features: {dataset_info['total_features']}")
    print("=" * 70)

    # Initialize seed for reproducibility
    init_seed(config.seed)

    # Prepare data
    print("Loading and preparing data...")

    available_datasets = [ds['file'] for ds in dataset_info['datasets']]

    dataset_file = 'Dataset/combined_dataframe_NYSE.csv'

    if not os.path.exists('Dataset'):
        print("❌ Dataset folder not found!")
        print("Please create a 'Dataset' folder and put your CSV files in it.")
        print("Expected files:")
        for ds in available_datasets:
            print(f"  - Dataset/{ds}")
        return

    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found!")
        print("Available datasets in Dataset folder:")
        for ds in available_datasets:
            full_path = f"Dataset/{ds}"
            if os.path.exists(full_path):
                print(f"  ✅ {full_path}")
            else:
                print(f"  ❌ {full_path}")
        return

    train_loader, val_loader, test_loader, mmn, num_features = prepare_data(
        csv_file=dataset_file,
        window=config.lag,
        predict=config.horizon,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio
    )

    config.num_nodes = num_features
    print(f"Number of features (graph nodes): {num_features}")

    args = config.to_dict()

    print("Initializing SAMBA + BTMF model...")
    model_args.vocab_size = num_features

    model = SAMBA_BTMF(
        model_args,
        args.get('hid'),
        args.get('lag'),
        args.get('horizon'),
        args.get('embed_dim'),
        args.get("cheb_k"),
        num_nodes=num_features
    )

    model = model.cuda()

    for name, p in model.named_parameters():
        if name.startswith('btmf.') or name == 'gamma':
            continue
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

    print(f"Final Results :")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Information Coefficient (IC): {IC:.4f}")
    print(f"Rank Information Coefficient (RIC): {RIC:.4f}")

    with torch.no_grad():
        bl = trainer.model.btmf if hasattr(trainer.model, 'BTMF') else model.btmf


    result_train_file = os.path.join("SAMBA_Model", "results")
    os.makedirs(result_train_file, exist_ok=True)

    with open('samba_BTMF_results.txt', 'a') as f:
        f.write(f"IC: {np.array(IC)}\n")
        f.write(f"RIC: {np.array(RIC)}\n")
        f.write(f"MAE: {np.array(mae)}\n")
        f.write(f"RMSE: {np.array(rmse)}\n\n")



if __name__ == "__main__":
    main()
