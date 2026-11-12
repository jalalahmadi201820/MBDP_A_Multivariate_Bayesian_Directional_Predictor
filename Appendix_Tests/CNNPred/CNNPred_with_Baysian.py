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
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_dropout(m):

    if type(m) == torch.nn.Dropout:
        m.train()


def bayesian_validate(args, model, dataset, train_prior_prob=None, mc_samples=10):

    model.eval()

    if mc_samples > 1:
        model.apply(apply_dropout)

    loss_fcn = torch.nn.BCELoss()
    data_dataloader = generate_batches(dataset, args.batch_size, n_workers=args.num_workers)

    loss_list = []
    pred_list = []
    label_list = []

    if train_prior_prob is not None:
        smoothed_prior = (train_prior_prob * 0.8) + (0.5 * 0.2)
        prior_pos = smoothed_prior
        prior_neg = 1.0 - smoothed_prior

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

            if train_prior_prob is not None:
                numerator = batch_mean_prob * prior_pos
                denominator = (batch_mean_prob * prior_pos) + ((1 - batch_mean_prob) * prior_neg)
                final_prob = numerator / (denominator + 1e-8)
            else:
                final_prob = batch_mean_prob
            # -------------------------------------------------------

            pred = (final_prob > 0.5).int()

            pred_list.extend(pred.cpu().numpy())
            label_list.extend(batch_label.cpu().numpy())
            loss_list.append(loss.item())

        loss_data = np.array(loss_list).mean()
        acc = accuracy(pred_list, label_list)
        f1 = f1_score(pred_list, label_list, average='macro')

    return loss_data, acc, f1


def train(args, train_dataset, val_dataset, test_dataset, i, prior_probability):
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
            # train_f1 = f1_score(pred_list, label_list, average='macro') 

            if (epoch + 1) % 10 == 0:
                print("Epoch {:03d} | Train Loss: {:.4f} | Train Acc: {:.4f}".format(epoch + 1, loss_data, train_acc))

            scheduler.step(loss_data)

            if (epoch + 1) % 1 == 0:
                val_loss, val_acc, val_f1 = bayesian_validate(args, model, val_dataset, prior_probability, mc_samples=1)

                if best_f1 < val_f1:
                    # print(f"--> New Best Validation F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
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

    print('\n=== RESULTS OF BEST MODEL (Using Smart Bayesian Inference) ===')

    MC_SAMPLES = 20

    train_loss, train_acc, train_f1 = bayesian_validate(args, model, train_dataset, prior_probability,
                                                        mc_samples=MC_SAMPLES)
    print("Train (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(train_loss, train_acc, train_f1))

    val_loss, val_acc, val_f1 = bayesian_validate(args, model, val_dataset, prior_probability, mc_samples=MC_SAMPLES)
    print("Valid (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(val_loss, val_acc, val_f1))

    test_loss, test_acc, test_f1 = bayesian_validate(args, model, test_dataset, prior_probability,
                                                     mc_samples=MC_SAMPLES)
    print("Test  (Bayesian MC):  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}".format(test_loss, test_acc, test_f1))

    print('---------------')
    return model


def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results, prior_probability):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]

        _, _, f1 = bayesian_validate(args, model, test_data, prior_probability, mc_samples=20)
        cnn_results.append(f1)

    return cnn_results


def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(args.num_iter - 1, len(order_stocks))
    cnn_results = pd.DataFrame(cnn_results, columns=order_stocks)

    summary = pd.DataFrame([cnn_results.mean(), cnn_results.max(), cnn_results.std()])
    cnn_results = pd.concat([cnn_results, summary], ignore_index=True)

    overall_mean_f1 = cnn_results.iloc[:args.num_iter-1].values.mean()
    print(f"\nAverage F1 over ALL iterations and ALL stocks: {overall_mean_f1:.4f}")

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

    cnn_results = []

    for i in range(1, args.num_iter):
        print('Iteration {}'.format(i))
        model = train(args, train_data, val_data, test_data, i, prior_probability)
        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results,
                                 prior_probability)

    saving_results(args, cnn_results, order_stocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNpred')
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='', help="Location of Base Directory")

    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=60, help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1, help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=50, help="number of repeating algorithm")

    parser.add_argument("--num-filter", type=int, default=8, help="number filters in conv layer")

    parser.add_argument("--drop", type=float, default=0.4, help="Increased dropout for MC sampling")

    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")

    parser.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay")

    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--patience', type=int, default=30, help="used for early stop")
    parser.add_argument('--override', default=True, type=bool, help='overrride the existing models')
    parser.add_argument('--num-workers', default=0, type=int)
    args = parser.parse_args()

    main(args)
