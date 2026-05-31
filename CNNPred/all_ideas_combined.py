from os.path import join
import argparse
import numpy as np
import pandas as pd
import os
import torch
import warnings
from pathlib import Path
from scipy.stats import ttest_ind
from CNNpred2D import CNNpred
from processing_data import costruct_data_warehouse, cnn_data_sequence, transforming_data_warehouse
from dataset import WholeDataset, generate_batches
from sklearn.metrics import accuracy_score as accuracy, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------
# Feature engineering: rolling skew & kurtosis
# ---------------------------------------------------------
def add_rolling_skew_kurt_to_dataframe(df, window=7):
    """
    Add rolling skewness and kurtosis features for all numeric columns.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {'label', 'target', 'Target', 'Label'}
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in numeric_cols:
        df[f'{col}_skew_{window}'] = df[col].rolling(window=window, min_periods=window).skew()
        df[f'{col}_kurt_{window}'] = df[col].rolling(window=window, min_periods=window).kurt()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def add_rolling_features_to_warehouse(data_warehouse, window=7):
    new_data_warehouse = {}
    for stock_name, df in data_warehouse.items():
        if isinstance(df, pd.DataFrame):
            new_data_warehouse[stock_name] = add_rolling_skew_kurt_to_dataframe(df, window=window)
        else:
            new_data_warehouse[stock_name] = df
    return new_data_warehouse


# ---------------------------------------------------------
# Bayesian utilities (Beta-Bernoulli + threshold tuning)
# ---------------------------------------------------------
def estimate_beta_prior_from_labels(labels, prior_strength=10.0):
    """
    Estimate Beta prior using empirical positive rate with pseudo-counts.
    labels: numpy array of 0/1
    prior_strength: total pseudo-counts (alpha+beta)
    """
    labels = np.asarray(labels).astype(np.float32)
    p_hat = labels.mean() if len(labels) > 0 else 0.5

    # pseudo-counts (Jeffreys-like safe clamp)
    alpha = max(1e-6, p_hat * prior_strength)
    beta = max(1e-6, (1.0 - p_hat) * prior_strength)

    return alpha, beta


def beta_prior_mean(alpha, beta):
    return alpha / (alpha + beta)


def calibrate_probs_with_beta_prior(probs, alpha, beta, lam=0.15):
    """
    probs: numpy array of model probabilities in [0,1]
    lam: mixing weight with prior mean
    """
    prior_m = beta_prior_mean(alpha, beta)
    calibrated = (1.0 - lam) * probs + lam * prior_m
    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def find_best_threshold(y_true, y_prob, thr_min=0.05, thr_max=0.95, thr_step=0.01):
    """
    Find threshold that maximizes macro F1 on validation set.
    """
    best_thr = 0.5
    best_f1 = -1.0

    thresholds = np.arange(thr_min, thr_max + 1e-12, thr_step)
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    return best_thr, best_f1


# ---------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------
def predict_probabilities(model, dataset, batch_size, num_workers):
    """
    Returns:
      probs: np.array of shape [N]
      labels: np.array of shape [N]
      avg_loss: float
    """
    model.eval()
    loss_fcn = torch.nn.BCELoss()
    data_dataloader = generate_batches(dataset, batch_size, n_workers=num_workers)

    probs_list = []
    labels_list = []
    loss_list = []

    with torch.no_grad():
        for batch_data, batch_label in data_dataloader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device).float()

            batch_prob = model(batch_data).view(-1)
            loss = loss_fcn(batch_prob, batch_label)

            probs_list.extend(batch_prob.cpu().numpy())
            labels_list.extend(batch_label.cpu().numpy())
            loss_list.append(loss.item())

    return np.array(probs_list), np.array(labels_list), float(np.mean(loss_list))


def validate_with_calibration(args, model, dataset, alpha, beta, threshold, lam):
    probs, labels, loss_data = predict_probabilities(
        model, dataset, args.batch_size, args.num_workers
    )

    calibrated_probs = calibrate_probs_with_beta_prior(probs, alpha, beta, lam=lam)
    pred = (calibrated_probs > threshold).astype(int)

    acc = accuracy(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    return loss_data, acc, f1


def validate_plain(args, model, dataset):
    probs, labels, loss_data = predict_probabilities(
        model, dataset, args.batch_size, args.num_workers
    )
    pred = (probs > 0.5).astype(int)

    acc = accuracy(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    return loss_data, acc, f1


# ---------------------------------------------------------
# Feature selection
# ---------------------------------------------------------
def perform_feature_selection(cnn_train_data, cnn_train_target, n_features, p_threshold=0.05):
    selected_features = []
    for feat in range(n_features):
        pos_data = cnn_train_data[cnn_train_target == 1, :, feat].flatten()
        neg_data = cnn_train_data[cnn_train_target == 0, :, feat].flatten()

        if len(pos_data) == 0 or len(neg_data) == 0:
            continue

        _, p_val = ttest_ind(pos_data, neg_data, equal_var=False)
        if p_val < p_threshold:
            selected_features.append(feat)

    selected_features = np.array(selected_features)
    print(f"Selected {len(selected_features)} features out of {n_features} based on t-test (p < {p_threshold})")
    return selected_features


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train(args, train_dataset, val_dataset, test_dataset, i):
    my_file = Path(join(args.Base_dir, f'2D-models/best-beta-{args.epochs}-{args.seq_len}-{args.num_filter}-{args.drop}-{i}.pt'))
    filepath = str(my_file)

    # --- Beta prior from train labels ---
    train_labels = np.asarray(train_dataset.label if hasattr(train_dataset, 'label') else train_dataset.target)
    alpha, beta = estimate_beta_prior_from_labels(train_labels, prior_strength=args.prior_strength)
    print(f"[Bayesian Prior] alpha={alpha:.4f}, beta={beta:.4f}, mean={beta_prior_mean(alpha,beta):.4f}")

    if my_file.is_file() and args.override is False:
        print('loading existing model...')
        model = CNNpred(args.number_feature, args.num_filter, args.drop).to(device)
        model.load_state_dict(torch.load(filepath, map_location=device))
    else:
        model = CNNpred(args.number_feature, args.num_filter, args.drop).to(device)

        best_f1 = -1
        cur_step = 0

        loss_fcn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, threshold=0.001,
            threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08, verbose=True
        )

        for epoch in range(args.epochs):
            model.train()
            loss_list, pred_list, label_list = [], [], []
            train_dataloader = generate_batches(train_dataset, args.batch_size, n_workers=args.num_workers)

            for batch_data, batch_label in train_dataloader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device).float()

                batch_prob = model(batch_data).view(-1)
                loss = loss_fcn(batch_prob, batch_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = (batch_prob > 0.5).int()
                pred_list.extend(pred.detach().cpu().numpy())
                label_list.extend(batch_label.detach().cpu().numpy())
                loss_list.append(loss.item())

            train_loss = float(np.mean(loss_list))
            train_acc = accuracy(label_list, pred_list)
            train_f1 = f1_score(label_list, pred_list, average='macro')

            print(f"Epoch {epoch+1:05d}")
            print(f"Train: loss={train_loss:.4f} | acc={train_acc:.4f} | f1={train_f1:.4f}")

            scheduler.step(train_loss)

            # --- validation with Bayesian calibration ---
            val_probs, val_labels, val_loss = predict_probabilities(
                model, val_dataset, args.batch_size, args.num_workers
            )
            val_probs_cal = calibrate_probs_with_beta_prior(val_probs, alpha, beta, lam=args.calibration_lambda)
            best_thr, val_best_f1 = find_best_threshold(
                val_labels, val_probs_cal,
                thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step
            )

            val_pred = (val_probs_cal > best_thr).astype(int)
            val_acc = accuracy(val_labels, val_pred)

            print(f"Validation: loss={val_loss:.4f} | acc={val_acc:.4f} | f1={val_best_f1:.4f} | best_thr={best_thr:.3f}")

            if val_best_f1 > best_f1:
                best_f1 = val_best_f1
                cur_step = 0
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                torch.save(model.state_dict(), filepath)
            else:
                cur_step += 1
                if cur_step >= args.patience:
                    print("Early stopping triggered.")
                    break

        # load best
        model.load_state_dict(torch.load(filepath, map_location=device))

    model = model.to(device)
    model.eval()

    # finalize best threshold on validation set
    val_probs, val_labels, _ = predict_probabilities(model, val_dataset, args.batch_size, args.num_workers)
    val_probs_cal = calibrate_probs_with_beta_prior(val_probs, alpha, beta, lam=args.calibration_lambda)
    best_thr, best_val_f1 = find_best_threshold(
        val_labels, val_probs_cal,
        thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step
    )

    print("\nresults of best model (Bayesian calibrated):")
    train_loss, train_acc, train_f1 = validate_with_calibration(
        args, model, train_dataset, alpha, beta, best_thr, args.calibration_lambda
    )
    print(f"Train: loss={train_loss:.4f} | acc={train_acc:.4f} | f1={train_f1:.4f}")

    val_loss, val_acc, val_f1 = validate_with_calibration(
        args, model, val_dataset, alpha, beta, best_thr, args.calibration_lambda
    )
    print(f"Validation: loss={val_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f} | thr={best_thr:.3f}")

    test_loss, test_acc, test_f1 = validate_with_calibration(
        args, model, test_dataset, alpha, beta, best_thr, args.calibration_lambda
    )
    print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f} | f1={test_f1:.4f}")
    print('---------------')

    return model, alpha, beta, best_thr


# ---------------------------------------------------------
# Prediction over all stocks
# ---------------------------------------------------------
def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results, alpha, beta, threshold):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]  # مطابق کد خودت
        _, _, f1 = validate_with_calibration(
            args, model, test_data, alpha, beta, threshold, args.calibration_lambda
        )
        cnn_results.append(f1)
    return cnn_results


def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results).reshape(args.num_iter, len(order_stocks))

    mean_results = cnn_results.mean(axis=0)
    max_results = cnn_results.max(axis=0)
    std_results = cnn_results.std(axis=0)

    cnn_df = pd.DataFrame(cnn_results, columns=order_stocks)
    summary_df = pd.DataFrame({
        'Mean (iterations)': mean_results,
        'Max': max_results,
        'Std': std_results
    }, index=order_stocks).T

    os.makedirs(join(args.Base_dir, '2D-models'), exist_ok=True)
    cnn_df.to_csv(join(args.Base_dir, '2D-models/detailed_results_bayesian_beta.csv'), index=True)
    summary_df.to_csv(join(args.Base_dir, '2D-models/summary_results_bayesian_beta.csv'))

    print("\n===== Summary =====")
    print(pd.DataFrame(mean_results, index=order_stocks, columns=['Mean F1']).T)
    print(f"BEST mean F1: {mean_results.max():.4f}")
    print(f"Average mean F1: {mean_results.mean():.4f}")
    print(f"Average std: {std_results.mean():.4f}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
    TRAIN_ROOT_PATH = join(args.Base_dir, 'Dataset')
    if not os.path.exists(TRAIN_ROOT_PATH):
        print(f"Error: Dataset path not found at {TRAIN_ROOT_PATH}")
        return

    train_file_names = os.listdir(TRAIN_ROOT_PATH)

    print('Loading train data ...')
    data_warehouse, number_of_stocks, args.number_feature, samples_in_each_stock = \
        costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names, args.predict_day, args.seq_len)

    print(f'number of stocks = {number_of_stocks}')
    print(f'number of original features = {args.number_feature}')
    print(f'number of samples in each stock = {samples_in_each_stock}')

    print(f'Adding rolling skewness and kurtosis with window={args.rolling_window} ...')
    data_warehouse = add_rolling_features_to_warehouse(data_warehouse, window=args.rolling_window)

    first_key = list(data_warehouse.keys())[0]
    first_df = data_warehouse[first_key]
    if isinstance(first_df, pd.DataFrame):
        numeric_cols = first_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = {'label', 'target', 'Target', 'Label'}
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        args.number_feature = len(feature_cols)

    print(f'number of features after FE = {args.number_feature}')

    order_stocks = data_warehouse.keys()
    transformed_data_loader_warehouse = transforming_data_warehouse(data_warehouse, order_stocks, args.seq_len)

    cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
        data_warehouse, args.seq_len
    )

    selected_features = perform_feature_selection(
        cnn_train_data, cnn_train_target, args.number_feature, p_threshold=args.p_threshold
    )

    if len(selected_features) == 0:
        print("Warning: No features selected. Using all features.")
        selected_features = np.arange(args.number_feature)

    cnn_train_data = cnn_train_data[:, :, selected_features]
    cnn_valid_data = cnn_valid_data[:, :, selected_features]
    cnn_test_data = cnn_test_data[:, :, selected_features]
    args.number_feature = len(selected_features)

    # apply same feature mask to transformed warehouse datasets
    for key in transformed_data_loader_warehouse:
        datasets = transformed_data_loader_warehouse[key]
        for ds in datasets:
            if hasattr(ds, 'data') and isinstance(ds.data, np.ndarray) and ds.data.ndim == 3:
                ds.data = ds.data[:, :, selected_features]
            elif hasattr(ds, 'data') and isinstance(ds.data, torch.Tensor) and ds.data.dim() == 3:
                ds.data = ds.data[:, :, selected_features]

    train_data = WholeDataset(cnn_train_data, cnn_train_target)
    val_data = WholeDataset(cnn_valid_data, cnn_valid_target)
    test_data = WholeDataset(cnn_test_data, cnn_test_target)

    cnn_results = []

    for i in range(args.num_iter):
        print(f'\n=== Iteration {i + 1}/{args.num_iter} ===')
        model, alpha, beta, threshold = train(args, train_data, val_data, test_data, i)
        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results, alpha, beta, threshold)

    saving_results(args, cnn_results, order_stocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNpred + Bayesian Beta Calibration')

    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='', help="Location of Base Directory")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=60, help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1, help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=10, help="number of repeating algorithm")
    parser.add_argument("--num-filter", type=int, default=8, help="number filters in conv layer")
    parser.add_argument("--drop", type=float, default=0.1, help="Fully connected dropout")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight decay")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--patience', type=int, default=80, help="early stop patience")
    parser.add_argument('--p_threshold', type=float, default=0.2, help="p-value threshold for t-test feature selection")
    parser.add_argument('--rolling_window', type=int, default=7, help='window size for rolling features')
    parser.add_argument('--override', default=True, type=bool, help='override existing models')
    parser.add_argument('--num-workers', default=0, type=int)

    # Bayesian calibration params
    parser.add_argument('--prior_strength', type=float, default=10.0,
                        help='pseudo-count strength for Beta prior (alpha+beta)')
    parser.add_argument('--calibration_lambda', type=float, default=0.15,
                        help='mixing weight between model prob and Beta prior mean')
    parser.add_argument('--thr_min', type=float, default=0.05, help='min threshold for search')
    parser.add_argument('--thr_max', type=float, default=0.95, help='max threshold for search')
    parser.add_argument('--thr_step', type=float, default=0.01, help='threshold search step')

    args = parser.parse_args()
    main(args)
