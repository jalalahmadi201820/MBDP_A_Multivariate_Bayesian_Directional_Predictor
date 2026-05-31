
from scipy.stats import skew, kurtosis as scipy_kurtosis

import os
import random
import pickle
import numpy as np
from scipy.stats import ttest_ind  # NEW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from Model import *
from Layers import *
from utils import *

CFG = dict(
    device_id="0",
    save_model=True,
    model_dir="./SavedModels",
    train_holdout=100,
    eval_size=50,
    rnn_length=20,
    d_hidden_factor=2,
    hidn_rnn=78,
    n_heads=2,
    hidn_att=39,
    dropout=0.3,
    t_mix=1,
    lr=8e-4,
    weight_decay=9.8e-4,
    clip_grad=0.45,
    batch_train=15,
    max_epoch=400,
    wait_epoch=30,
    seed=1021,
    mc_eval=30,
    kl_beta=None,
    num_stock_override=73,
    # NEW: t-test config
    ttest_alpha=0.05,      # threshold for p-value
    ttest_min_samples=10,  # minimum samples per class for each feature
    save_selected_features=True
)


class FlexibleIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, σ_prior=1.0, ρ_init=-5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.02))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), ρ_init))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), ρ_init))

        self.register_buffer('σ_prior', torch.tensor(float(σ_prior)))

    @staticmethod
    def _softplus(x):
        return torch.log1p(torch.exp(x))

    def _sample(self, mu, rho):
        eps = torch.randn_like(rho)
        sigma = self._softplus(rho)
        return mu + sigma * eps, sigma

    def _kl(self, mu_q, σ_q):
        σ_p = self.σ_prior
        return torch.sum(torch.log(σ_p / σ_q) + (σ_q ** 2 + mu_q ** 2) / (2 * σ_p ** 2) - 0.5)

    def forward(self, x, sample=True):
        if self.training or sample:
            w, σ_w = self._sample(self.weight_mu, self.weight_rho)
            b, σ_b = self._sample(self.bias_mu, self.bias_rho)
            kl = self._kl(self.weight_mu, self._softplus(self.weight_rho)) + \
                 self._kl(self.bias_mu, self._softplus(self.bias_rho))
        else:
            w = self.weight_mu
            b = self.bias_mu
            kl = torch.tensor(0., device=x.device)

        return F.linear(x, w, b), kl


class GraphCNN_Bayes(GraphCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_out = kwargs.get('out_c', 2)

        detected_in = 117

        print("\n" + "=" * 60)
        print("[GraphCNN_Bayes] Initializing Structure Fix & Dimension Check...")

        if hasattr(self, 'X2Os'):
            target_layer = self.X2Os
            print(f" >> Inspecting target layer 'X2Os' for input dimension...")

            found_dim = False
            for name, param in target_layer.named_parameters():
                if param.dim() >= 2:
                    if param.shape[0] == self.target_out:
                        detected_in = param.shape[1]
                        found_dim = True
                        print(f"    -> Found weight {tuple(param.shape)}. Detected Input Dim: {detected_in}")
                        break
                    elif param.shape[1] == self.target_out:
                        detected_in = param.shape[0]
                        found_dim = True
                        print(f"    -> Found weight {tuple(param.shape)}. Detected Input Dim: {detected_in}")
                        break

            if not found_dim:
                print(f"    -> Could not infer dim from weights. Using fallback: {detected_in}")

            print(" >> Replacing 'self.X2Os' with FlexibleIdentity...")
            self.X2Os = FlexibleIdentity()

        elif hasattr(self, 'fc'):
            print(" >> Found 'fc' layer. Replacing...")
            if hasattr(self.fc, 'in_features'):
                detected_in = self.fc.in_features
            self.fc = FlexibleIdentity()

        else:
            print(" >> WARNING: Target layer not found. Assuming input dim is correct.")

        print(f" >> Initializing Bayesian Layer: In={detected_in}, Out={self.target_out}")
        self.bayes_fc = BayesianLinear(detected_in, self.target_out)
        print("=" * 60 + "\n")

    def forward(self, mkt, news, edge_list, inter_metric, device, sample=True):
        h_raw = super().forward(mkt, news, edge_list, inter_metric, device)

        if isinstance(h_raw, tuple):
            h_raw = h_raw[0]

        if h_raw.dim() > 2:
            h_raw = h_raw.flatten(start_dim=1)

        if h_raw.shape[-1] != self.bayes_fc.in_features:
            print(f"\n!! CRITICAL MISMATCH DETECTED !!")
            print(f"Tensor shape: {h_raw.shape}, Layer expects: {self.bayes_fc.in_features}")
            print(f"Attempting dynamic fix (Re-initializing layer)...")
            self.bayes_fc = BayesianLinear(h_raw.shape[-1], self.target_out).to(h_raw.device).to(h_raw.dtype)

        logits, kl = self.bayes_fc(h_raw, sample=sample)
        return F.log_softmax(logits, dim=-1), kl

def add_rolling_skew_kurt(x_tensor, window=7):
    """
    x_tensor: [T, N, D]
    returns: [T, N, D*3]  (original + rolling_skew + rolling_kurt)
    """
    x_np = x_tensor.cpu().numpy()  # [T, N, D]
    T, N, D = x_np.shape
    skew_arr = np.zeros_like(x_np)
    kurt_arr = np.zeros_like(x_np)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # suppress catastrophic cancellation
        for d in range(D):
            for t in range(T):
                start = max(0, t - window + 1)
                window_data = x_np[start:t+1, :, d]  # [w, N]
                if window_data.shape[0] < 2:
                    continue
                skew_arr[t, :, d] = skew(window_data, axis=0, nan_policy='omit')
                kurt_arr[t, :, d] = scipy_kurtosis(window_data, axis=0, nan_policy='omit')

    # FIX: replace NaN/inf with 0 (happens when window data is nearly constant)
    skew_arr = np.nan_to_num(skew_arr, nan=0.0, posinf=0.0, neginf=0.0)
    kurt_arr = np.nan_to_num(kurt_arr, nan=0.0, posinf=0.0, neginf=0.0)

    result = np.concatenate([x_np, skew_arr, kurt_arr], axis=2)  # [T, N, D*3]
    return torch.tensor(result, dtype=x_tensor.dtype, device=x_tensor.device)

def load_dataset(device):
    base_path = './data/'
    with open(base_path + 'x_num_standard.pkl', 'rb') as f: markets = pickle.load(f)
    with open(base_path + 'y_1.pkl', 'rb') as f: y_load = pickle.load(f)
    with open(base_path + 'x_newtext.pkl', 'rb') as f: sentiments = pickle.load(f)
    with open(base_path + 'edge_new.pkl', 'rb') as f: edge_list = pickle.load(f)
    with open(base_path + 'interactive.pkl', 'rb') as f: inter_metric = pickle.load(f)

    x = torch.tensor(markets, dtype=torch.double, device=device)
    y = torch.tensor(y_load, device=device).squeeze()
    y = (y > 0).to(torch.long)
    xs = torch.tensor(sentiments, dtype=torch.double, device=device)

    inter_metric = torch.tensor(inter_metric, device=device).squeeze(2).transpose(0, 1)
    return x, y, xs, edge_list, inter_metric


# NEW
def select_features_ttest_train_only(x_tr, y_tr, alpha=0.05, min_samples=10):
    """
    x_tr: [T, N, D]
    y_tr: ideally [T, N] but can be [T, N, 1] or [T, N, K]
    """
    x_np = x_tr.detach().cpu().numpy()
    y_np = y_tr.detach().cpu().numpy()

    # --- robust shape handling for y ---
    # remove trivial singleton dims first
    y_np = np.squeeze(y_np)

    if y_np.ndim == 1:
        # maybe flattened labels -> try to infer [T, N]
        T, N, D = x_np.shape
        if y_np.size == T * N:
            y_np = y_np.reshape(T, N)
        else:
            raise ValueError(f"[TTEST] y_tr is 1D with size={y_np.size}, cannot match x_tr shape={x_np.shape}")

    elif y_np.ndim == 2:
        # expected case [T, N]
        pass

    elif y_np.ndim >= 3:
        # if last dim has class channels or extra fields, pick first channel
        # You can change to y_np[..., -1] if your valid label is last channel
        y_np = y_np[..., 0]
        y_np = np.squeeze(y_np)
        if y_np.ndim != 2:
            raise ValueError(f"[TTEST] After squeezing y_tr, expected 2D got shape={y_np.shape}")

    T_x, N_x, D = x_np.shape
    T_y, N_y = y_np.shape

    # align T and N safely (avoid crash)
    T = min(T_x, T_y)
    N = min(N_x, N_y)

    if (T != T_x) or (N != N_x) or (T != T_y) or (N != N_y):
        print(f"[TTEST][WARN] shape mismatch fixed by cropping: "
              f"x_tr {x_np.shape} -> {(T, N, D)}, y_tr {y_np.shape} -> {(T, N)}")

    x_np = x_np[:T, :N, :]
    y_np = y_np[:T, :N]

    # ensure binary labels 0/1
    y_np = (y_np > 0).astype(np.int64)

    x_flat = x_np.reshape(T * N, D)
    y_flat = y_np.reshape(T * N)

    cls0 = (y_flat == 0)
    cls1 = (y_flat == 1)

    pvals = np.ones(D, dtype=np.float64)
    mean_diff = np.zeros(D, dtype=np.float64)

    for d in range(D):
        v0 = x_flat[cls0, d]
        v1 = x_flat[cls1, d]

        v0 = v0[np.isfinite(v0)]
        v1 = v1[np.isfinite(v1)]

        if len(v0) < min_samples or len(v1) < min_samples:
            pvals[d] = 1.0
            mean_diff[d] = 0.0
            continue

        mean_diff[d] = np.mean(v1) - np.mean(v0)

        _, p = ttest_ind(v1, v0, equal_var=False, nan_policy='omit')
        if np.isnan(p):
            p = 1.0
        pvals[d] = p

    selected_idx = np.where(pvals < alpha)[0]

    if len(selected_idx) == 0:
        print("[TTEST][WARN] no feature passed threshold. fallback -> keep all features.")
        selected_idx = np.arange(D)

    stats = {
        "pvals": pvals,
        "mean_diff": mean_diff,
        "num_total": D,
        "num_selected": len(selected_idx),
        "alpha": alpha
    }
    return selected_idx, stats


def apply_feature_selection(x_tensor, selected_idx):
    idx_t = torch.tensor(selected_idx, dtype=torch.long, device=x_tensor.device)
    return x_tensor.index_select(dim=2, index=idx_t)


def train_epoch(model, x_tr, xs_tr, y_tr, edges, inter_metric, device,
                cfg, optimizer, criterion, β):
    model.train()
    seq_idx = list(range(len(x_tr)))[cfg['rnn_length']:]
    random.shuffle(seq_idx)
    total_nll = total_kl = 0.0

    optimizer.zero_grad()
    for step, i in enumerate(seq_idx):
        mkt_batch = x_tr[i - cfg['rnn_length'] + 1: i + 1]
        news_batch = xs_tr[i - cfg['rnn_length'] + 1: i + 1]

        logp, kl = model(mkt_batch, news_batch, edges, inter_metric, device, sample=True)

        target = y_tr[i][:cfg['num_stock_override']]
        nll = criterion(logp, target)

        loss = nll + β * kl
        loss.backward()

        total_nll += nll.item()
        total_kl += kl.item()

        if (step % cfg['batch_train']) == (cfg['batch_train'] - 1):
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()
            optimizer.zero_grad()

    if len(seq_idx) % cfg['batch_train']:
        nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
        optimizer.step()

    return total_nll / len(seq_idx), total_kl / len(seq_idx)


@torch.no_grad()
def evaluate(model, x_ev, xs_ev, y_ev, edges, inter_metric, device, cfg, mc_samples):
    model.eval()
    seq_idx = list(range(len(x_ev)))[cfg['rnn_length']:]
    preds, trues = [], []

    for i in seq_idx:
        mkt_batch = x_ev[i - cfg['rnn_length'] + 1: i + 1]
        news_batch = xs_ev[i - cfg['rnn_length'] + 1: i + 1]

        probs_mc = []
        for _ in range(mc_samples):
            logp, _ = model(mkt_batch, news_batch, edges, inter_metric, device, sample=True)
            probs_mc.append(logp.exp().cpu().numpy())

        probs = np.mean(probs_mc, axis=0)
        preds.append(probs)
        trues.append(y_ev[i][:cfg['num_stock_override']].cpu().numpy())

    return metrics(trues, preds)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    cfg = CFG
    device = "cuda:" + cfg['device_id'] if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    set_seed(cfg['seed'])

    print("Loading data...")
    x, y, xs, edge_list, inter_metric = load_dataset(device)
    x = add_rolling_skew_kurt(x, window=7)

    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = xs.size(2)

    rnn_len = cfg['rnn_length']
    idx_tr = -cfg['train_holdout']
    idx_ev = -cfg['eval_size']

    x_tr = x[:idx_tr]
    x_ev = x[idx_tr - rnn_len: idx_ev]
    x_te = x[idx_ev - rnn_len:]

    xs_tr = xs[:idx_tr]
    xs_ev = xs[idx_tr - rnn_len: idx_ev]
    xs_te = xs[idx_ev - rnn_len:]

    y_tr = y[:idx_tr]
    y_ev = y[idx_tr - rnn_len: idx_ev]
    y_te = y[idx_ev - rnn_len:]

    # ---------------- t-test feature selection on train only ----------------
    print("\nRunning train-only t-test feature selection on market features...")
    selected_idx, fs_stats = select_features_ttest_train_only(
        x_tr=x_tr,
        y_tr=y_tr,
        alpha=cfg['ttest_alpha'],
        min_samples=cfg['ttest_min_samples']
    )

    print(f"[T-TEST] Selected {fs_stats['num_selected']} / {fs_stats['num_total']} features "
          f"(alpha={fs_stats['alpha']})")

    x_tr = apply_feature_selection(x_tr, selected_idx)
    x_ev = apply_feature_selection(x_ev, selected_idx)
    x_te = apply_feature_selection(x_te, selected_idx)

    if cfg.get('save_selected_features', False):
        os.makedirs(cfg['model_dir'], exist_ok=True)
        np.save(os.path.join(cfg['model_dir'], 'selected_feature_idx.npy'), selected_idx)
        np.save(os.path.join(cfg['model_dir'], 'ttest_pvals.npy'), fs_stats['pvals'])
        np.save(os.path.join(cfg['model_dir'], 'ttest_mean_diff.npy'), fs_stats['mean_diff'])
        print(f"[T-TEST] Saved selected indices & stats in: {cfg['model_dir']}")
    # ---------------------------------------------------------------------------

    NUM_STOCK = x_tr.size(1)
    D_MARKET = x_tr.size(2)
    D_NEWS = xs_tr.size(2)

    print(f"Stocks: {NUM_STOCK}, Market(after t-test): {D_MARKET}, News: {D_NEWS}")

    model = GraphCNN_Bayes(
        num_stock=NUM_STOCK,
        d_market=D_MARKET,
        d_news=D_NEWS,
        out_c=2,
        d_hidden=D_MARKET * cfg['d_hidden_factor'],
        hidn_rnn=cfg['hidn_rnn'],
        hid_c=cfg['hidn_att'],
        n_heads=cfg['n_heads'],
        dropout=cfg['dropout'],
        t_mix=cfg['t_mix']
    ).to(device).double()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = nn.NLLLoss()

    N_train = len(x_tr) - rnn_len
    β = cfg['kl_beta'] if cfg['kl_beta'] is not None else 1.0 / N_train

    print("Start Training...")
    best_auc = 0.0
    best_path = None
    wait = 0

    for epoch in range(cfg['max_epoch']):
        nll, kl = train_epoch(model, x_tr, xs_tr, y_tr, edge_list, inter_metric,
                              device, cfg, optimizer, criterion, β)

        acc_e, auc_e = evaluate(model, x_ev, xs_ev, y_ev, edge_list, inter_metric, device, cfg, cfg['mc_eval'])
        acc_t, auc_t = evaluate(model, x_te, xs_te, y_te, edge_list, inter_metric, device, cfg, cfg['mc_eval'])

        print(
            f"Epoch {epoch:03d} | NLL: {nll:.4f} | KL: {kl:.4f} | "
            f"Val AUC: {auc_e:.4f} | Val ACC: {acc_e:.4f} | Test AUC: {auc_t:.4f}"
        )

        if auc_e > best_auc:
            best_auc = auc_e
            wait = 0
            if cfg['save_model']:
                os.makedirs(cfg['model_dir'], exist_ok=True)
                if best_path:
                    try:
                        os.remove(best_path)
                    except:
                        pass
                best_path = os.path.join(cfg['model_dir'], f"best_bayes_auc{auc_e:.4f}.pth")
                torch.save(model.state_dict(), best_path)
        else:
            wait += 1

        if wait >= cfg['wait_epoch']:
            print(f"Early stopping! Best Val AUC: {best_auc:.4f}")
            break


if __name__ == "__main__":
    main()
