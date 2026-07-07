from Model import *
import pickle
from Layers import *
from torch import optim
import argparse
import torch
import torch.nn as nn


parser = argparse.ArgumentParser()

parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='400',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='8e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='2',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='39',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='0.00098',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='20',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.3',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.45',
                    help='rnn clip')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')

parser.add_argument('--alpha-h', type=float, default='0.2',
                    help='Bayesian hidden state update coefficient (init)')
parser.add_argument('--alpha-c', type=float, default='0.1',
                    help='Bayesian cell state forget coefficient (init)')
parser.add_argument('--k-up', type=float, default='1.0',
                    help='Bayesian likelihood-up scaling (init)')
parser.add_argument('--k-down', type=float, default='1.0',
                    help='Bayesian likelihood-down scaling (init)')
parser.add_argument('--bay-forget', type=float, default='0.9',
                    help='Bayesian prior forgetting factor (init)')



def load_dataset(device1):
    with open('data/x_num_standard.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('data/y_1.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('data/x_newtext.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open('data/edge_new.pkl', 'rb') as handle:
        edge_list = pickle.load(handle)
    with open('data/interactive.pkl', 'rb') as handle:
        interactive_metric = pickle.load(handle)
    print(len(markets))
    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=device1)
    x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=device1)
    x_sentiment.to(torch.double)
    y = torch.tensor(y_load, device=device1).squeeze()
    y = (y > 0).to(torch.long)
    inter_metric = torch.tensor(interactive_metric, device=device1)
    inter_metric = inter_metric.squeeze(2)
    inter_metric = inter_metric.transpose(0, 1)
    return x, y, x_sentiment, edge_list, inter_metric



class BTMF(nn.Module):


    def __init__(self, num_stock, d_market, hidden=2):
        super(BTMF, self).__init__()
        self.num_stock = num_stock
        self.d_market = d_market
        self.hidden = hidden

        self.raw_alpha_h = nn.Parameter(torch.tensor(self._inv_sigmoid(0.2)))
        self.raw_alpha_c = nn.Parameter(torch.tensor(self._inv_sigmoid(0.1)))
        self.raw_k_up = nn.Parameter(torch.tensor(1.0))
        self.raw_k_down = nn.Parameter(torch.tensor(1.0))
        self.raw_bay_forget = nn.Parameter(torch.tensor(self._inv_sigmoid(0.9)))


        self.feature_proj = nn.Linear(d_market, 4)


        self.mix_gate = nn.Parameter(torch.tensor(0.5))

        self.register_buffer('h1', torch.zeros(num_stock, dtype=torch.double))
        self.register_buffer('h2', torch.zeros(num_stock, dtype=torch.double))
        self.register_buffer('c1', torch.zeros(num_stock, dtype=torch.double))
        self.register_buffer('c2', torch.zeros(num_stock, dtype=torch.double))
        self.register_buffer('priorUp', torch.full((num_stock,), 0.5, dtype=torch.double))
        self.register_buffer('priorDown', torch.full((num_stock,), 0.5, dtype=torch.double))

    @staticmethod
    def _inv_sigmoid(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    def reset_state(self, num_stock=None, device=None, dtype=torch.double):
        n = num_stock if num_stock is not None else self.num_stock
        dev = device if device is not None else self.h1.device
        self.h1 = torch.zeros(n, dtype=dtype, device=dev)
        self.h2 = torch.zeros(n, dtype=dtype, device=dev)
        self.c1 = torch.zeros(n, dtype=dtype, device=dev)
        self.c2 = torch.zeros(n, dtype=dtype, device=dev)
        self.priorUp = torch.full((n,), 0.5, dtype=dtype, device=dev)
        self.priorDown = torch.full((n,), 0.5, dtype=dtype, device=dev)

    def _params(self):
        alpha_h = torch.sigmoid(self.raw_alpha_h)
        alpha_c = torch.sigmoid(self.raw_alpha_c)
        bay_forget = torch.sigmoid(self.raw_bay_forget)
        k_up = torch.nn.functional.softplus(self.raw_k_up)
        k_down = torch.nn.functional.softplus(self.raw_k_down)
        return alpha_h, alpha_c, k_up, k_down, bay_forget

    def update_state(self, short_ret, long_ret):
        alpha_h, alpha_c, _, _, _ = self._params()

        x1 = short_ret
        x2 = long_ret

        h1_new = (1.0 - alpha_h) * self.h1 + alpha_h * x1
        h2_new = (1.0 - alpha_h) * self.h2 + alpha_h * x2

        c1_new = (1.0 - alpha_c) * self.c1 + alpha_c * h1_new
        c2_new = (1.0 - alpha_c) * self.c2 + alpha_c * h2_new

        self.h1 = h1_new.detach()
        self.h2 = h2_new.detach()
        self.c1 = c1_new.detach()
        self.c2 = c2_new.detach()

        return h1_new, h2_new, c1_new, c2_new

    def update_posterior(self, short_ret, long_ret, rsi, volatility):
        _, _, k_up, k_down, bay_forget = self._params()

        priorUp = bay_forget * self.priorUp + (1.0 - bay_forget) * 0.5
        priorDown = bay_forget * self.priorDown + (1.0 - bay_forget) * 0.5

        sign_short = torch.sign(short_ret)
        sign_long = torch.sign(long_ret)

        scoreUp = sign_short + sign_long
        scoreUp = scoreUp + torch.where(rsi < 70.0,
                                         torch.full_like(rsi, 0.5),
                                         torch.full_like(rsi, -0.5))
        scoreUp = scoreUp - volatility * 0.2

        scoreDown = -sign_short - sign_long
        scoreDown = scoreDown + torch.where(rsi > 30.0,
                                             torch.full_like(rsi, 0.3),
                                             torch.full_like(rsi, 0.8))
        scoreDown = scoreDown + volatility * 0.2

        likeUp = torch.clamp(1.0 + k_up * scoreUp, min=0.01)
        likeDown = torch.clamp(1.0 + k_down * scoreDown, min=0.01)

        postUp = likeUp * priorUp
        postDown = likeDown * priorDown

        norm = postUp + postDown
        norm = torch.clamp(norm, min=1e-8)

        newUp = postUp / norm
        newDown = postDown / norm

        safe_mask = (postUp + postDown) <= 0.0
        newUp = torch.where(safe_mask, torch.full_like(newUp, 0.5), newUp)
        newDown = torch.where(safe_mask, torch.full_like(newDown, 0.5), newDown)

        self.priorUp = newUp.detach()
        self.priorDown = newDown.detach()

        return newUp, newDown

    def forward(self, market_slice):

        feats = self.feature_proj(market_slice.to(torch.double))
        short_ret = feats[:, 0]
        long_ret = feats[:, 1]
        rsi = torch.sigmoid(feats[:, 2]) * 100.0
        volatility = torch.nn.functional.softplus(feats[:, 3])

        self.update_state(short_ret, long_ret)
        up, down = self.update_posterior(short_ret, long_ret, rsi, volatility)

        probs = torch.stack([down, up], dim=1)
        probs = torch.clamp(probs, min=1e-8)
        log_probs = torch.log(probs)
        return log_probs

    def confidence_gate(self):

        confidence = torch.clamp(self.priorUp - self.priorDown, min=0.0)
        return confidence


class GraphCNNWithBTMF(nn.Module):
    def __init__(self, base_model, num_stock, d_market):
        super(GraphCNNWithBTMF, self).__init__()
        self.base_model = base_model
        self.bay = BTMF(num_stock=num_stock, d_market=d_market)

    def reset_bay_state(self):
        self.bay.reset_state()

    def forward(self, x_window, x_sentiment_window, edge_list, inter_metric, device1):
        gnn_log_probs = self.base_model(x_window, x_sentiment_window, edge_list, inter_metric, device1)

        last_day_market = x_window[-1]
        bay_log_probs = self.bay(last_day_market)

        gate = torch.sigmoid(self.bay.mix_gate)
        gnn_probs = torch.exp(gnn_log_probs)
        bay_probs = torch.exp(bay_log_probs)

        mixed_probs = gate * gnn_probs + (1.0 - gate) * bay_probs
        mixed_probs = torch.clamp(mixed_probs, min=1e-8)
        mixed_log_probs = torch.log(mixed_probs)

        return mixed_log_probs


def train(model, x_train, x_sentiment_train, y_train, edge_list, inter_metric, device1):
    model.train()
    model.reset_bay_state()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 15
    for i in train_seq:
        output = model(x_train[i - rnn_length + 1: i + 1], x_sentiment_train[i - rnn_length + 1: i + 1], edge_list,
                        inter_metric, device1)
        loss = criterion(output, y_train[i][:73])
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count


def evaluate(model, x_eval, x_sentiment_eval, y_eval, edge_list, device1):
    model.eval()
    model.reset_bay_state()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    with torch.no_grad():
        for i in seq:
            output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], edge_list,
                            inter_metric, device1)
            output = output.detach().cpu()
            preds.append(np.exp(output.numpy()))
            trues.append(y_eval[i][:73].cpu().numpy())
    acc, auc = metrics(trues, preds)
    return acc, auc


if __name__ == "__main__":
    args = parser.parse_args()
    device1 = "cuda:" + args.device
    device1 = device1
    print(device1)
    criterion = torch.nn.NLLLoss()
    set_seed(1021)
    # load dataset
    print("loading dataset")
    x, y, x_sentiment, edge_list, inter_metric = load_dataset(device1)
    print(len(x), len(y))
    # hyper-parameters
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    MAX_EPOCH = args.max_epoch
    hidn_rnn = args.hidn_rnn
    N_heads = args.heads_att
    hidn_att = args.hidn_att
    lr = args.lr
    rnn_length = args.rnn_length
    t_mix = 1
    edge_list = edge_list

    # train-valid-test split
    x_train = x[: -100]
    x_eval = x[-100 - rnn_length: -50]
    x_test = x[-50 - rnn_length:]

    y_train = y[: -100]
    y_eval = y[-100 - rnn_length: -50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[: -100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length: -50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]

    ## initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    base_model = GraphCNN(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS, out_c=2,
                           d_hidden=D_MARKET * 2, hidn_rnn=hidn_rnn, hid_c=hidn_att, n_heads=N_heads,
                           dropout=args.dropout, t_mix=t_mix)

    model = GraphCNNWithBTMF(base_model=base_model, num_stock=NUM_STOCK, d_market=D_MARKET)

    model.cuda(device=device1)
    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

    # train
    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train, x_sentiment_train, y_train, edge_list, inter_metric, device1)
        eval_acc, eval_auc = evaluate(model, x_eval, x_sentiment_eval, y_eval, edge_list, device1)
        test_acc, test_auc = evaluate(model, x_test, x_sentiment_test, y_test, edge_list, device1)
        eval_str = "epoch{},train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(
            epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
        print(eval_str)

        if eval_acc > eval_epoch_best:
            eval_epoch_best = eval_acc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/best_model_auc{:.4f}.pth".format(eval_auc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch >= 50:
            print("saved_model_result:", eval_best_str)
            break
        epoch += 1
