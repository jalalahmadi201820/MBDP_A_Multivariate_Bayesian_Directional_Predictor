
import os
import random
import pickle
import numpy as np
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



def load_dataset(device):
    base_path = './data/'
    with open(base_path + 'x_num_standard.pkl', 'rb') as f:   markets = pickle.load(f)
    with open(base_path + 'y_1.pkl', 'rb') as f:   y_load = pickle.load(f)
    with open(base_path + 'x_newtext.pkl', 'rb') as f:   sentiments = pickle.load(f)
    with open(base_path + 'edge_new.pkl', 'rb') as f:   edge_list = pickle.load(f)
    with open(base_path + 'interactive.pkl', 'rb') as f:   inter_metric = pickle.load(f)

    x = torch.tensor(markets, dtype=torch.double, device=device)
    y = torch.tensor(y_load, device=device).squeeze()
    y = (y > 0).to(torch.long)
    xs = torch.tensor(sentiments, dtype=torch.double, device=device)

    inter_metric = torch.tensor(inter_metric, device=device).squeeze(2).transpose(0, 1)
    return x, y, xs, edge_list, inter_metric



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

    print(f"Stocks: {NUM_STOCK}, Market: {D_MARKET}, News: {D_NEWS}")

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
            f"Epoch {epoch:03d} | NLL: {nll:.4f} | KL: {kl:.4f} | Val AUC: {auc_e:.4f} | Val ACC: {acc_e:.4f} | Test AUC: {auc_t:.4f}")

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
