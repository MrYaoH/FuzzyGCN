import logging, argparse
from torch_geometric import datasets

from regression_eval import *


OUT_PATH = 'result/'

# parser for hyperparameters
parser = argparse.ArgumentParser()
dataset_name = 'cora'
parser.add_argument('--dataset', type=str, default=dataset_name, help='{cora, pubmed, citeseer}.')
parser.add_argument('--dataset_dir', type=str, default='./dataset/' + dataset_name)
parser.add_argument('--model', type=str, default='FGCN', help='{GCN, FGCN}')
parser.add_argument('--hid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.001, help='Number of hidden units.')
parser.add_argument('--epochs', type=int, default=400, help='Number of hidden units.')
parser.add_argument('--log', type=str, default='debug', help='Number of hidden units.')
parser.add_argument('--wd', type=float, default=1.1e-2, help='Number of hidden units.')
parser.add_argument('--alpha', type=float, default=9.e-4, help='Number of hidden units.')

parser.add_argument('--num_eval_splits', type=int, default=10, help='Number of hidden units.')

flag_parser = parser.add_mutually_exclusive_group(required=False)

args = parser.parse_args()
logging.basicConfig(format="%(message)s", level=getattr(logging, args.log.upper()))

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.Planetoid('./dataset/cora', name="Cora")
num_eval_splits = args.num_eval_splits
data = dataset[0]
data = data.to(device)

data_split = dataset_split(data.y, device, train_ratio=20, val_ratio=500, test_ratio=1000)
ans = eval_model(args, data, data_split, device, logging)

print('Test_ACC:', ans)
print('fgcn')
