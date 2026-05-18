from os.path import join
import argparse
import numpy as np
import pandas as pd
import os
import torch
import warnings
from pathlib import Path
import tqdm

from CNNpred2D import CNNpred
from processing_data import costruct_data_warehouse, cnn_data_sequence_pre_train, cnn_data_sequence, \
    transforming_data_warehouse
from dataset import WholeDataset, generate_batches
from sklearn.metrics import accuracy_score as accuracy, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_ohlc_rolling_features(data_warehouse):
    target_cols = ['Open', 'High', 'Low', 'Close', 'open', 'high', 'low', 'close']
    features_added = 0
    window_size = 5

    print(f" ${window_size}$")

    for key, data in data_warehouse.items():
        if isinstance(data, pd.DataFrame):
            found_cols = [col for col in data.columns if col in target_cols]

            if len(found_cols) > 0 and f"{found_cols[0]}_skew_10" not in data.columns:
                for col in found_cols:
                    data[f'{col}_skew_10'] = data[col].rolling(window=window_size).skew().fillna(0)
                    data[f'{col}_kurt_10'] = data[col].rolling(window=window_size).kurt().fillna(0)

                features_added = len(found_cols) * 2
    return data_warehouse, features_added


def validate(args, model, dataset):
    model.eval()
    loss_fcn = torch.nn.BCELoss()

    data_dataloader = generate_batches(dataset, args.batch_size, n_workers=args.num_workers)

    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch_data, batch_label in data_dataloader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device).float()

            batch_logit = model(batch_data).view(-1)
            loss = loss_fcn(batch_logit, batch_label)
            pred = (batch_logit > 0.5).int()

            pred_list.extend(pred.cpu().numpy())
            label_list.extend(batch_label.cpu().numpy())
            loss_list.append(loss.item())

        loss_data = np.array(loss_list).mean()
        acc = accuracy(pred_list, label_list)
        f1 = f1_score(pred_list, label_list, average='macro')

    return loss_data, acc, f1


def train(args, train_dataset, val_dataset, test_dataset, i):
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
                                                               patience=20,
                                                               threshold=0.001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0.00001,
                                                               eps=1e-08,
                                                               verbose=True)

        start_epoch = 0

        for epoch in range(start_epoch, args.epochs):
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
                parameters = list(model.parameters())
                optimizer.step()
                loss_list.append(loss.item())

            loss_data = np.array(loss_list).mean()
            train_acc = accuracy(pred_list, label_list)
            train_f1 = f1_score(pred_list, label_list, average='macro')

            print("Epoch {:05d}\n"
                  "Train: loss: {:.4f} | accuracy: {:.4f} | f-acore: {:.4f}"
                  .format(epoch + 1, loss_data, train_acc, train_f1))

            scheduler.step(loss_data)

            if (epoch + 1) % 1 == 0:
                val_loss, val_acc, val_f1 = validate(args, model, val_dataset)
                print("Validation:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
                      .format(val_loss, val_acc, val_f1))

                if best_f1 < val_f1:
                    best_f1 = val_f1
                    cur_step = 0
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    torch.save(model, filepath)
                else:
                    cur_step += 1
                    if cur_step == args.patience:
                        break

    try:
        model = torch.load(filepath, weights_only=False)
    except TypeError:
        model = torch.load(filepath)

    model = model.to(device)

    print('results of best model')

    train_loss, train_acc, train_f1 = validate(args, model, train_dataset)
    print("Train:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(train_loss, train_acc, train_f1))

    val_loss, val_acc, val_f1 = validate(args, model, val_dataset)
    print("Validation:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(val_loss, val_acc, val_f1))

    test_loss, test_acc, test_f1 = validate(args, model, test_dataset)
    print("Test:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(test_loss, test_acc, test_f1))

    print('---------------')

    return model, (train_loss, train_acc, train_f1), (val_loss, val_acc, val_f1), (test_loss, test_acc, test_f1)


def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]
        cnn_results.append(validate(args, model, test_data)[2])

    return cnn_results


def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(args.num_iter - 1, len(order_stocks))
    cnn_results = pd.DataFrame(cnn_results, columns=order_stocks)

    summary = pd.DataFrame([cnn_results.mean(), cnn_results.max(), cnn_results.std()])
    cnn_results = pd.concat([cnn_results, summary], ignore_index=True)

    os.makedirs(join(args.Base_dir, '2D-models'), exist_ok=True)
    cnn_results.to_csv(join(args.Base_dir, '2D-models/new results.csv'), index=False)


def main(args):
    TRAIN_ROOT_PATH = join(args.Base_dir, 'Dataset')
    if not os.path.exists(TRAIN_ROOT_PATH):
        print(f"Error: Dataset path not found at {TRAIN_ROOT_PATH}")
        return

    train_file_names = os.listdir(join(args.Base_dir, 'Dataset'))

    print('Loading train data ...')
    data_warehouse, number_of_stocks, args.number_feature, samples_in_each_stock = \
        costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names, args.predict_day, args.seq_len)

    data_warehouse, added_feats = add_ohlc_rolling_features(data_warehouse)


    if added_feats > 0:
        args.number_feature += added_feats

    order_stocks = data_warehouse.keys()
    transformed_data_loader_warehouse = transforming_data_warehouse(data_warehouse, order_stocks, args.seq_len)

    cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
        data_warehouse, args.seq_len)

    train_data = WholeDataset(cnn_train_data, cnn_train_target)
    val_data = WholeDataset(cnn_valid_data, cnn_valid_target)
    test_data = WholeDataset(cnn_test_data, cnn_test_target)

    cnn_results = []

    all_train_results = []
    all_val_results = []
    all_test_results = []

    for i in range(1, args.num_iter):
        print('Iteration {}'.format(i))
        model, train_metrics, val_metrics, test_metrics = train(args, train_data, val_data, test_data, i)

        all_train_results.append(train_metrics)
        all_val_results.append(val_metrics)
        all_test_results.append(test_metrics)

        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results)

    saving_results(args, cnn_results, order_stocks)

    return all_train_results, all_val_results, all_test_results


def run_multiple_times(num_runs=5):
    parser = argparse.ArgumentParser(description='CNNpred')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='',
                        help="Location of Base Directory")
    parser.add_argument("--epochs", type=int, default=150,
                        help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1,
                        help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=2,
                        help="number of repeating algorithm")
    parser.add_argument("--num-filter", type=int, default=8,
                        help="number filters in conv layer")
    parser.add_argument("--drop", type=float, default=0.1,
                        help="Fully connected dropout")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=200,
                        help="used for early stop")
    parser.add_argument('--override', default=True, type=bool,
                        help='overrride the existing models')
    parser.add_argument('--num-workers', default=0, type=int)

    args = parser.parse_args()

    all_runs_train = []
    all_runs_val = []
    all_runs_test = []

    for run_idx in range(num_runs):
        print(f" {run_idx + 1} از {num_runs}")

        train_results, val_results, test_results = main(args)

        all_runs_train.append(train_results)
        all_runs_val.append(val_results)
        all_runs_test.append(test_results)

    print(f" {num_runs} ")

    for iter_idx in range(len(all_runs_train[0])):
        train_losses = [run[iter_idx][0] for run in all_runs_train]
        train_accs = [run[iter_idx][1] for run in all_runs_train]
        train_f1s = [run[iter_idx][2] for run in all_runs_train]

        val_losses = [run[iter_idx][0] for run in all_runs_val]
        val_accs = [run[iter_idx][1] for run in all_runs_val]
        val_f1s = [run[iter_idx][2] for run in all_runs_val]

        test_losses = [run[iter_idx][0] for run in all_runs_test]
        test_accs = [run[iter_idx][1] for run in all_runs_test]
        test_f1s = [run[iter_idx][2] for run in all_runs_test]

        print(f"\nIteration {iter_idx + 1}:")
        print("-" * 80)
        print(f"Train (Average):  loss: {np.mean(train_losses):.4f} (±{np.std(train_losses):.4f}) | "
              f"accuracy: {np.mean(train_accs):.4f} (±{np.std(train_accs):.4f}) | "
              f"f1: {np.mean(train_f1s):.4f} (±{np.std(train_f1s):.4f})")

        print(f"Validation (Average):  loss: {np.mean(val_losses):.4f} (±{np.std(val_losses):.4f}) | "
              f"accuracy: {np.mean(val_accs):.4f} (±{np.std(val_accs):.4f}) | "
              f"f1: {np.mean(val_f1s):.4f} (±{np.std(val_f1s):.4f})")

        print(f"Test (Average):  loss: {np.mean(test_losses):.4f} (±{np.std(test_losses):.4f}) | "
              f"accuracy: {np.mean(test_accs):.4f} (±{np.std(test_accs):.4f}) | "
              f"f1: {np.mean(test_f1s):.4f} (±{np.std(test_f1s):.4f})")


if __name__ == '__main__':
    run_multiple_times(num_runs=200)
