from os.path import join
import argparse
import numpy as np
import pandas as pd
import os
import torch
import warnings
import random
from pathlib import Path
from CNNpred2D import CNNpred
from processing_data import costruct_data_warehouse, cnn_data_sequence, \
    transforming_data_warehouse
from dataset import WholeDataset, generate_batches
from sklearn.metrics import accuracy_score as accuracy, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()



class BTMF:
    def __init__(self, alpha_h=0.2, alpha_c=0.1, k_up=1.0, k_down=1.0):
        self.alpha_h = alpha_h
        self.alpha_c = alpha_c
        self.k_up = k_up
        self.k_down = k_down
        self.reset()

    def reset(self):
        self.h1 = 0.0
        self.h2 = 0.0
        self.c1 = 0.0
        self.c2 = 0.0
        self.prior_up = 0.5
        self.prior_down = 0.5

    def update_state(self, short_ret, long_ret, rsi, volatility):
        x1 = short_ret
        x2 = long_ret

        self.h1 = (1.0 - self.alpha_h) * self.h1 + self.alpha_h * x1
        self.h2 = (1.0 - self.alpha_h) * self.h2 + self.alpha_h * x2

        self.c1 = (1.0 - self.alpha_c) * self.c1 + self.alpha_c * self.h1
        self.c2 = (1.0 - self.alpha_c) * self.c2 + self.alpha_c * self.h2

        self.prior_up = 0.9 * self.prior_up + 0.1 * 0.5
        self.prior_down = 0.9 * self.prior_down + 0.1 * 0.5

    def update_posterior(self, short_ret, long_ret, rsi, volatility):
        score_up = 0.0
        score_down = 0.0

        score_up += 1.0 if short_ret > 0 else -1.0
        score_up += 1.0 if long_ret > 0 else -1.0
        score_up += 0.5 if rsi < 70.0 else -0.5
        score_up -= volatility * 0.2

        score_down -= 1.0 if short_ret > 0 else -1.0
        score_down -= 1.0 if long_ret > 0 else -1.0
        score_down += 0.3 if rsi > 30.0 else 0.8
        score_down += volatility * 0.2

        like_up = max(0.01, 1.0 + self.k_up * score_up)
        like_down = max(0.01, 1.0 + self.k_down * score_down)

        post_up = like_up * self.prior_up
        post_down = like_down * self.prior_down

        norm = post_up + post_down
        if norm <= 0.0:
            self.prior_up = 0.5
            self.prior_down = 0.5
        else:
            self.prior_up = post_up / norm
            self.prior_down = post_down / norm

    def prob_up(self):
        return self.prior_up

    def prob_down(self):
        return self.prior_down


    def step_from_probs(self, batch_mean_prob_np, prev_prob=None, rsi=50.0, volatility=0.01):
        n = len(batch_mean_prob_np)
        out_prior_up = np.zeros(n, dtype=np.float64)

        for idx in range(n):
            cur_prob = float(batch_mean_prob_np[idx])
            if prev_prob is None:
                short_ret = cur_prob - 0.5
                long_ret = cur_prob - 0.5
            else:
                short_ret = cur_prob - prev_prob
                long_ret = cur_prob - 0.5

            self.update_state(short_ret, long_ret, rsi, volatility)
            self.update_posterior(short_ret, long_ret, rsi, volatility)

            out_prior_up[idx] = self.prior_up
            prev_prob = cur_prob

        return out_prior_up, prev_prob


def bayesian_validate(args, model, dataset, train_prior_prob=None, mc_samples=10, bay_module=None):
    model.eval()

    if mc_samples > 1:
        model.apply(apply_dropout)

    loss_fcn = torch.nn.BCELoss()
    data_dataloader = generate_batches(dataset, args.batch_size, n_workers=args.num_workers)

    loss_list = []
    pred_list = []
    label_list = []

    use_static_prior = (train_prior_prob is not None) and (bay_module is None)
    if use_static_prior:
        smoothed_prior = (train_prior_prob * 0.8) + (0.5 * 0.2)
        prior_pos = smoothed_prior
        prior_neg = 1.0 - smoothed_prior

    if bay_module is not None:
        bay_module.reset()
        prev_prob_state = None

    with torch.no_grad():
        for batch_data, batch_label in data_dataloader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device).float()

            mc_outputs = []
            for _ in range(mc_samples):
                out = model(batch_data).view(-1)
                mc_outputs.append(out)

            batch_mean_prob = torch.stack(mc_outputs).mean(dim=0)

            loss = loss_fcn(batch_mean_prob, batch_label)

            if bay_module is not None:
                batch_mean_prob_np = batch_mean_prob.cpu().numpy()
                mc_stack = torch.stack(mc_outputs)
                volatility_est = mc_stack.std(dim=0).cpu().numpy()

                prior_up_arr = np.zeros_like(batch_mean_prob_np)
                for idx in range(len(batch_mean_prob_np)):
                    cur_prob = float(batch_mean_prob_np[idx])
                    vol = float(volatility_est[idx])
                    if prev_prob_state is None:
                        short_ret = cur_prob - 0.5
                    else:
                        short_ret = cur_prob - prev_prob_state
                    long_ret = cur_prob - 0.5
                    rsi_proxy = cur_prob * 100.0

                    bay_module.update_state(short_ret, long_ret, rsi_proxy, vol)
                    bay_module.update_posterior(short_ret, long_ret, rsi_proxy, vol)

                    prior_up_arr[idx] = bay_module.prob_up()
                    prev_prob_state = cur_prob

                prior_pos_t = torch.tensor(prior_up_arr, dtype=batch_mean_prob.dtype, device=device)
                prior_neg_t = 1.0 - prior_pos_t

                numerator = batch_mean_prob * prior_pos_t
                denominator = (batch_mean_prob * prior_pos_t) + ((1 - batch_mean_prob) * prior_neg_t)
                final_prob = numerator / (denominator + 1e-8)

            elif use_static_prior:
                numerator = batch_mean_prob * prior_pos
                denominator = (batch_mean_prob * prior_pos) + ((1 - batch_mean_prob) * prior_neg)
                final_prob = numerator / (denominator + 1e-8)
            else:
                final_prob = batch_mean_prob

            pred = (final_prob > 0.5).int()

            pred_list.extend(pred.cpu().numpy())
            label_list.extend(batch_label.cpu().numpy())
            loss_list.append(loss.item())

        loss_data = np.array(loss_list).mean()
        acc = accuracy(pred_list, label_list)
        f1 = f1_score(pred_list, label_list, average='macro')

    return loss_data, acc, f1


def train(args, train_dataset, val_dataset, test_dataset, i, prior_probability, bay_module):
    my_file = Path(join(args.Base_dir,
                        '2D-models/best-{}-{}-{}-{}-{}.pt'.format(args.epochs, args.seq_len, args.num_filter, args.drop,
                                                                  i)))
    filepath = join(args.Base_dir,
                    '2D-models/best-{}-{}-{}-{}-{}.pt'.format(args.epochs, args.seq_len, args.num_filter, args.drop, i))

    if my_file.is_file() and args.override == False:
        print('loading model')
    else:
        model = CNNpred(args.number_feature, args.num_filter, args.drop).to(device)
        cur_step = 0
        best_f1 = -1

        loss_fcn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=10)

        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            pred_list = []
            label_list = []
            train_dataloader = generate_batches(train_dataset, args.batch_size, n_workers=args.num_workers)

            for batch_data, batch_label in train_dataloader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device).float()

                batch_logit = model(batch_data).view(-1)
                loss = loss_fcn(batch_logit, batch_label)

                pred = (batch_logit > 0.5).int()
                pred_list.extend(pred.cpu().numpy())
                label_list.extend(batch_label.cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            loss_data = np.array(loss_list).mean()
            train_acc = accuracy(pred_list, label_list)
            if (epoch + 1) % 10 == 0:
                print("Epoch {:03d} | Train Loss: {:.4f} | Train Acc: {:.4f}".format(epoch + 1, loss_data, train_acc))

            scheduler.step(loss_data)

            if (epoch + 1) % 1 == 0:
                val_loss, val_acc, val_f1 = bayesian_validate(args, model, val_dataset, prior_probability,
                                                              mc_samples=1, bay_module=bay_module)

                if best_f1 < val_f1:
                    best_f1 = val_f1
                    cur_step = 0
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    torch.save(model, filepath)
                else:
                    cur_step += 1
                    if cur_step >= args.patience:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break

    try:
        model = torch.load(filepath, weights_only=False)
    except TypeError:
        model = torch.load(filepath)

    model = model.to(device)

    print('\n=== RESULTS OF BEST MODEL  ===')

    MC_SAMPLES = 30

    train_loss, train_acc, train_f1 = bayesian_validate(args, model, train_dataset, prior_probability,
                                                        mc_samples=MC_SAMPLES, bay_module=bay_module)
    print("Train (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(train_loss, train_acc, train_f1))

    val_loss, val_acc, val_f1 = bayesian_validate(args, model, val_dataset, prior_probability,
                                                  mc_samples=MC_SAMPLES, bay_module=bay_module)
    print("Valid (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(val_loss, val_acc, val_f1))

    test_loss, test_acc, test_f1 = bayesian_validate(args, model, test_dataset, prior_probability,
                                                     mc_samples=MC_SAMPLES, bay_module=bay_module)
    print("Test  (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(test_loss, test_acc, test_f1))

    print('---------------')
    return model, test_f1


def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results, prior_probability, bay_module):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]

        _, _, f1 = bayesian_validate(args, model, test_data, prior_probability, mc_samples=20, bay_module=bay_module)
        cnn_results.append(f1)

    return cnn_results


def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(args.num_iter - 1, len(order_stocks))
    cnn_results = pd.DataFrame(cnn_results, columns=order_stocks)

    summary = pd.DataFrame([cnn_results.mean(), cnn_results.max(), cnn_results.std()])
    cnn_results = pd.concat([cnn_results, summary], ignore_index=True)

    overall_mean_f1 = cnn_results.iloc[:args.num_iter - 1].values.mean()
    print(f"\nAverage F1 over ALL iterations and ALL stocks: {overall_mean_f1:.4f}")

    os.makedirs(join(args.Base_dir, '2D-models'), exist_ok=True)
    cnn_results.to_csv(join(args.Base_dir, '2D-models/new results.csv'), index=False)


def main(args):
    set_seed(args.seed)

    TRAIN_ROOT_PATH = join(args.Base_dir, 'Dataset')
    if not os.path.exists(TRAIN_ROOT_PATH):
        print(f"Error: Dataset path not found at {TRAIN_ROOT_PATH}")
        return

    train_file_names = os.listdir(join(args.Base_dir, 'Dataset'))

    print('Loading train data ...')
    data_warehouse, number_of_stocks, args.number_feature, samples_in_each_stock = \
        costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names, args.predict_day, args.seq_len)

    order_stocks = data_warehouse.keys()
    transformed_data_loader_warehouse = transforming_data_warehouse(data_warehouse, order_stocks, args.seq_len)

    cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
        data_warehouse, args.seq_len)

    if isinstance(cnn_train_target, torch.Tensor):
        train_targets_np = cnn_train_target.cpu().numpy()
    else:
        train_targets_np = np.array(cnn_train_target)

    prior_probability = np.mean(train_targets_np)
    print(f"Global Train Prior: {prior_probability:.4f}")

    train_data = WholeDataset(cnn_train_data, cnn_train_target)
    val_data = WholeDataset(cnn_valid_data, cnn_valid_target)
    test_data = WholeDataset(cnn_test_data, cnn_test_target)

    bay_module = BTMF(alpha_h=0.3, alpha_c=0.2, k_up=1.0, k_down=1.0)

    cnn_results = []
    all_iterations_f1 = []

    for i in range(1, args.num_iter):
        print('Iteration {}'.format(i))
        model, test_f1 = train(args, train_data, val_data, test_data, i, prior_probability, bay_module)
        all_iterations_f1.append(test_f1)

        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results,
                                 prior_probability, bay_module)

    all_iterations_f1 = np.array(all_iterations_f1)
    mean_f1 = np.mean(all_iterations_f1)
    var_f1 = np.var(all_iterations_f1)

    print("\n" + "=" * 50)
    print("FINAL RESULTS ACROSS ALL ITERATIONS:")
    print(f"All F1 Scores: {all_iterations_f1.tolist()}")
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Variance F1: {var_f1:.6f}")
    print("=" * 50 + "\n")

    saving_results(args, cnn_results, order_stocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNpred')
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='', help="Location of Base Directory")

    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=58, help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1, help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=20, help="number of repeating algorithm")

    parser.add_argument("--num-filter", type=int, default=8, help="number filters in conv layer")

    parser.add_argument("--drop", type=float, default=0.3, help="Increased dropout for MC sampling")

    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")

    parser.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay")

    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--patience', type=int, default=30, help="used for early stop")
    parser.add_argument('--override', default=True, type=bool, help='overrride the existing models')
    parser.add_argument('--num-workers', default=0, type=int)

    parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')

    args = parser.parse_args()

    main(args)
