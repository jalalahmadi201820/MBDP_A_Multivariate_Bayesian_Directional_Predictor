

from os.path import join
import argparse
import numpy as np
import pandas as pd
import os
import torch
import warnings
from pathlib import Path
import tqdm
from scipy.stats import ttest_ind
from CNNpred2D import CNNpred
from processing_data import costruct_data_warehouse, cnn_data_sequence_pre_train, cnn_data_sequence, \
    transforming_data_warehouse
from dataset import WholeDataset, generate_batches
from sklearn.metrics import accuracy_score as accuracy, f1_score
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        model = CNNpred(args.number_feature, args.num_filter, args.drop).to(device)  # مدل باید اینجا هم به دیوایس برود
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

    return model


def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]

        cnn_results.append(validate(args, model, test_data)[2])

    return cnn_results


def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(args.num_iter, len(order_stocks))  
    
    mean_results = cnn_results.mean(axis=0)
    max_results = cnn_results.max(axis=0)
    std_results = cnn_results.std(axis=0)

    cnn_df = pd.DataFrame(cnn_results, columns=order_stocks)

    summary_df = pd.DataFrame({
        'Mean (10 iterations)': mean_results,
        'Max': max_results,
        'Std': std_results
    }, index=order_stocks).T

    os.makedirs(join(args.Base_dir, '2D-models'), exist_ok=True)

    cnn_df.to_csv(join(args.Base_dir, '2D-models/detailed_results_10_iterations.csv'), index=True)

    mean_df = pd.DataFrame(mean_results, index=order_stocks, columns=['Mean F1 (10 iterations)']).T
    mean_df.to_csv(join(args.Base_dir, '2D-models/mean_results_10_iterations.csv'), index=True)

    summary_df.to_csv(join(args.Base_dir, '2D-models/summary_results_10_iterations.csv'))

    print(mean_df)
    print(f"\n BEST: {mean_results.max():.4f}")
    print(f"Average: {mean_results.mean():.4f}")
    print(f"Std: {std_results.mean():.4f}")


def perform_feature_selection(cnn_train_data, cnn_train_target, n_features, p_threshold=0.05):
    """
    Perform t-test based feature selection using Welch's t-test for balanced importance of classes.
    Selects features with significant difference in means between positive and negative classes.
    """
    selected_features = []
    for feat in range(n_features):
        # Flatten all values for this feature across samples and timesteps, grouped by label
        pos_data = cnn_train_data[cnn_train_target == 1, :, feat].flatten()
        neg_data = cnn_train_data[cnn_train_target == 0, :, feat].flatten()

        if len(pos_data) == 0 or len(neg_data) == 0:
            continue

        # Use Welch's t-test (equal_var=False) to handle potentially unequal variances and sample sizes
        # This ensures balanced consideration of both classes regardless of count
        t_stat, p_val = ttest_ind(pos_data, neg_data, equal_var=False)

        if p_val < p_threshold:
            selected_features.append(feat)

    selected_features = np.array(selected_features)
    print(f"Selected {len(selected_features)} features out of {n_features} based on t-test (p < {p_threshold})")
    return selected_features


def main(args):
    TRAIN_ROOT_PATH = join(args.Base_dir, 'Dataset')
    if not os.path.exists(TRAIN_ROOT_PATH):
        print(f"Error: Dataset path not found at {TRAIN_ROOT_PATH}")
        return

    train_file_names = os.listdir(join(args.Base_dir, 'Dataset'))

    print('Loading train data ...')
    data_warehouse, number_of_stocks, args.number_feature, samples_in_each_stock = \
        costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names, args.predict_day, args.seq_len)

    print('number of stocks = {}'.format(number_of_stocks))
    print('number of features = {}'.format(args.number_feature))
    print('number of samples in each stock = {}'.format(samples_in_each_stock))

    order_stocks = data_warehouse.keys()
    transformed_data_loader_warehouse = transforming_data_warehouse(data_warehouse, order_stocks, args.seq_len)

    cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
        data_warehouse, args.seq_len)

    # Perform feature selection using t-test on training data
    # This selects features with significant association to labels, treating both classes equally important
    # (via Welch's t-test, independent of sample sizes)
    selected_features = perform_feature_selection(
        cnn_train_data, cnn_train_target, args.number_feature, p_threshold=args.p_threshold
    )

    if len(selected_features) == 0:
        print("Warning: No features selected. Using all features.")
        selected_features = np.arange(args.number_feature)

    # Slice the global datasets to selected features
    cnn_train_data = cnn_train_data[:, :, selected_features]
    cnn_valid_data = cnn_valid_data[:, :, selected_features]
    cnn_test_data = cnn_test_data[:, :, selected_features]

    # Update number of features
    args.number_feature = len(selected_features)

    # Slice the per-stock datasets in transformed_data_loader_warehouse
    # Assuming each value is a list/tuple of WholeDataset objects, e.g., [train, val, test]
    # and each dataset has a 'data' attribute (numpy array or tensor)
    for key in transformed_data_loader_warehouse:
        datasets = transformed_data_loader_warehouse[key]
        for ds in datasets:
            if hasattr(ds, 'data') and ds.data.ndim == 3:  # Assuming shape (n_samples, seq_len, n_features)
                ds.data = ds.data[:, :, selected_features]
            # If data is torch tensor, ensure it's sliced properly
            elif hasattr(ds, 'data') and isinstance(ds.data, torch.Tensor) and ds.data.dim() == 3:
                ds.data = ds.data[:, :, selected_features]

    train_data = WholeDataset(cnn_train_data, cnn_train_target)
    val_data = WholeDataset(cnn_valid_data, cnn_valid_target)
    test_data = WholeDataset(cnn_test_data, cnn_test_target)

    cnn_results = []

    for i in range(args.num_iter):
        print(f'\n=== Iteration {i + 1}/10 ===')
        model = train(args, train_data, val_data, test_data, i)
        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results)

    saving_results(args, cnn_results, order_stocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNpred')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='',
                        help="Location of Base Directory")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1,
                        help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=10,  
                        help="number of repeating algorithm (10 iterations)")
    parser.add_argument("--num-filter", type=int, default=8,
                        help="number filters in conv layer")
    parser.add_argument("--drop", type=float, default=0.1,
                        help="Fully connected dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,  
                        help="weight decay")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=200,
                        help="used for early stop")
    parser.add_argument('--p_threshold', type=float, default=0.2,
                        help="p-value threshold for t-test feature selection")

    parser.add_argument('--override', default=True, type=bool,
                        help='overrride the existing models')
    # -----------------------------------------------

    parser.add_argument('--num-workers', default=0, type=int)
    args = parser.parse_args()

    main(args)
