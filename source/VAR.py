import os
import sys
import argparse
import json

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

import torch
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from source.utils.utils import *
from source.utils.torchUtils import *
from source.layers.models import *
from source.layers.ablation_models import *
from source.utils.dataloader import *

# my args
my_data_path = '../data/enb2'
my_cache_file = '../data/cache.pickle'
my_model_type = 'heteroNRI'
my_test = False
my_graph_type = 'heteroNRI'  # or heteroNRI_gru
my_model_path = '../result/heteroNRI'
my_lag = 7

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--data_type', type=str, default='skt',
                    help='one of: skt')
parser.add_argument('--data_path', type=str, default=my_data_path)
parser.add_argument('--tr', type=float, default=0.7,
                    help='the ratio of training data to the original data')
parser.add_argument('--val', type=float, default=0.2,
                    help='the ratio of validation data to the original data')
parser.add_argument('--standardize', action='store_true',
                    help='standardize the inputs if it is true.')
parser.add_argument('--exclude_TA', action='store_true',
                    help='exclude TA column if it is set true.')
parser.add_argument('--lag', type=int, default=my_lag,
                    help='time-lag (default: 1)')
parser.add_argument('--cache_file', type=str, default=my_cache_file,
                    help='a cache file to min-max scale the data')
parser.add_argument('--graph_time_range', type=int, default=36,
                    help='time-range to save a graph')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--kl_loss_penalty', type=float, default=0.01, help='kl-loss penalty (default= 0.01)')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type=float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type=int, default=10, help='print training loss every print_log_option')
parser.add_argument('--verbose', action='store_true',
                    help='print logs about early-stopping')

# model options
parser.add_argument('--model_path', type=str, default=my_model_path,
                    help='a path to (save) the model')
parser.add_argument('--num_blocks', type=int, default=3,
                    help='the number of the HeteroBlocks (default= 3)')
parser.add_argument('--k', type=int, default=2,
                    help='the number of layers at every GC-Module (default= 2)')
parser.add_argument('--top_k', type=int, default=4,
                    help='top_k to select as non-zero in the adjacency matrix    (default= 4)')
parser.add_argument('--embedding_dim', type=int, default=128,
                    help='the size of embedding dimesion in the graph-learning layer (default= 128)')
parser.add_argument('--alpha', type=float, default=3.,
                    help='controls saturation rate of tanh: activation function in the graph-learning layer (default= 3.0)')
parser.add_argument('--beta', type=float, default=0.5,
                    help='parameter used in the GraphConvolutionModule, must be in the interval [0,1] (default= 0.5)')
# only for the heteroNRI, NRI
parser.add_argument('--tau', type=float, default=1.,
                    help='smoothing parameter used in the Gumbel-Softmax, only used in the model: heteroNRI')
# only fot eh NRI
parser.add_argument('--n_hid_encoder', type=int, default=256,
                    help='dimension of a hidden vector in the nri-encoder')
parser.add_argument('--msg_hid', type=int, default=256,
                    help='dimension of a message vector in the nri-decoder')
parser.add_argument('--msg_out', type=int, default=256,
                    help='dimension of a message vector (out) in the nri-decoder')
parser.add_argument('--n_hid_decoder', type=int, default=256,
                    help='dimension of a hidden vector in the nri-decoder')

# graphlearning type
parser.add_argument('--graph_type', type=str, default=my_graph_type,
                    help='decide graph learning type')

# To test
# parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--test', type=bool, default=my_test
                    , help='model file', required=False)
parser.add_argument('--model_file', type=str, default='latest_checkpoint.pth.tar'
                    , help='model file', required=False)
parser.add_argument('--model_type', type=str, default=my_model_type,
                    help='one of: \'mtgnn\', \'heteroNRI\'... ')

parser.add_argument('--num_folds', type=int, default=1,
                    help='the number of folds')

args = parser.parse_args()
print(args)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok=True)
else:
    print("The path already exists, skip making the path...")

print(f'saving the commandline arguments in the path: {args.model_path}...')
args_file = os.path.join(args.model_path, 'commandline_args.txt')
with open(args_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def main(args):
    # read data
    print("Loading data...")
    if args.data_type == 'skt':
        # load gestures-data
        data = load_skt(args) if not args.exclude_TA else load_skt_without_TA(args)
    else:
        print("Unkown data type, data type should be \"skt\"")
        sys.exit()





    # model
    data = torch.cat([data['train'][0],data['valid'][0],data['test'][0]],dim=1)
    total_preds=[]
    total_labels=[]
    te_r2 = 0
    te_mae = 0
    te_mse = 0

    for enb in range(len(data)):
        enb_data = data[0]
        enb_data = pd.DataFrame(enb_data, columns=args.columns)  # to DataFrame
        '''
        Make training set and validation set and test set.
        test set only has shape (obs-lag+1,lag,col) .
        training, validation set have shape (obs, lag, col).
        '''
        start_idx_val = int(enb_data.shape[0] * args.tr)
        start_idx_te = start_idx_val + int(enb_data.shape[0] * args.val)
        train = enb_data.iloc[:start_idx_val, :]
        valid = enb_data.iloc[start_idx_val:start_idx_te, :]
        test = enb_data.iloc[start_idx_te:, :]

        test_input = []
        test_label = []
        for idx in range(test.shape[0] - args.lag -1 + 1): #-1 is pred_steps
            input = test.iloc[idx:idx + args.lag, :]
            label = test.iloc[idx + args.lag, :]
            label = np.array(label).reshape(1,-1)
            test_input.append(input)
            test_label.append(label)

        #make VAR model
        model = VAR(train)
        results = model.fit(args.lag)

        preds = []
        for sample in test_input:
            out = results.forecast(sample.values,1)
            preds.append(out)


        preds = np.concatenate(preds)
        test_label = np.concatenate(test_label)

        r2 = r2_score(preds.flatten(),test_label.flatten())
        mae = mean_absolute_error(preds.flatten(),test_label.flatten())
        mse = mean_squared_error(preds.flatten(),test_label.flatten())

        total_preds.append(preds)
        total_labels.append(test_label)
        te_r2+= r2
        te_mae+= mae
        te_mse+= mse


    #get score
    te_r2 = te_r2/len(data)
    te_mae = te_mae/len(data)
    te_mse = te_mse/len(data)

    print(f"r2: {te_r2:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"mse: {te_mse:.2f}")
    print()



if __name__ == '__main__':
    if args.num_folds == 1:
        main(args)
    else:
        perf = main(args)
        perfs = dict().fromkeys(perf, None)
        for k in perfs.keys():
            perfs[k] = [perf[k]]

        for i in range(1, args.num_folds):
            perf = main(args)
            for k in perfs.keys():
                perfs[k].append(perf[k])

        for k, v in perfs.items():
            perfs[k] = [np.mean(perfs[k]), np.std(perfs[k])]

        for k, v in perfs.items():
            print(f"{k}: mean= {v[0]:.3f}, std= {v[1]:.3f}")