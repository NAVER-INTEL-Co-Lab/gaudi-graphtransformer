import habana_frameworks.torch.core as htcore
import argparse
import copy
import os
os.environ["PT_HPU_LAZY_MODE"] = "0"

import random
import sys
import warnings
import time, subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, eval_rocauc, evaluate, load_fixed_splits, class_rand_splits, to_sparse_tensor
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
import time

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
print(args)

# fix_seed(args.seed)

device = torch.device("hpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

dataset.label = dataset.label.to(device)

n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]

# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")


_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

if args.method == 'graphormer':
    dataset.graph['x'] = dataset.graph['x'].to(device)
    dataset.graph['in_degree'] = dataset.graph['in_degree'].to(device)
    dataset.graph['out_degree'] = dataset.graph['out_degree'].to(device)
    dataset.graph['spatial_pos'] = dataset.graph['spatial_pos'].to(device)
    dataset.graph['attn_bias'] = dataset.graph['attn_bias'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

# using rocauc as the eval function
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins','deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
# model = torch.compile(model, backend = "hpu_backend")
model = model.to(device)
print('MODEL:', model)

### Training loop ###
patience = 0

run_time_list = []

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    
    if (args.method == 'ours' or args.method == 'sgformer') and args.use_graph:
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.ours_weight_decay},
            {'params': model.params2, 'weight_decay': args.weight_decay}
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        emb = None
        if args.method == 'nodeformer':
            out, link_loss_ = model(dataset)
        else:
            out = model(dataset)
        
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            if args.method == 'graphormer':
                out = out.squeeze(0)
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
                
        if args.method == 'nodeformer':
            loss -= args.lamda * sum(link_loss_) / len(link_loss_)
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()
        run_time = 1000 * (end_time - start_time)
        run_time_list.append(run_time)

        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

run_time = sum(run_time_list) / len(run_time_list)
run_time_std = np.std(run_time_list)
results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    else:
        print_str += f'train_prop:{args.train_prop}, valid_prop:{args.valid_prop}'
    if method == 'ours':
        use_weight=' ours_use_weight' if args.ours_use_weight else ''
        print_str += f'method: {args.method} hidden: {args.hidden_channels} ours_layers:{args.ours_layers} lr:{args.lr} use_graph:{args.use_graph} aggregate:{args.aggregate} graph_weight:{args.graph_weight} alpha:{args.alpha} ours_decay:{args.ours_weight_decay} ours_dropout:{args.ours_dropout} epochs:{args.epochs} use_feat_norm:{not args.no_feat_norm} use_bn:{args.use_bn} use_residual:{args.ours_use_residual} use_act:{args.ours_use_act}{use_weight}\n'
        if not args.use_graph:
            return print_str
        if args.backbone == 'gcn':
            print_str += f'backbone:{args.backbone}, layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay} dropout:{args.dropout}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'
if args.method == 'ours' and args.use_graph:
    file_name += '_' + args.backbone
file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write(f' run_time: { run_time } ± {run_time_std}')
    f.write('\n\n')
