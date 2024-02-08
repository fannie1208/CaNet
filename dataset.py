from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, to_dense_adj, dense_to_sparse

from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv


import pickle as pkl
import os

class GCN_gen(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN_gen, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)
        return x

def load_twitch_dataset(data_dir, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    transform = T.NormalizeFeatures()
    sub_graphs = ['DE', 'PT', 'RU', 'ES', 'FR', 'EN']
    x_list, edge_index_list, y_list, env_list = [], [], [], []
    node_idx_list = []
    idx_shift = 0
    for i, g in enumerate(sub_graphs):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
                              name=g, transform=transform)
        data = torch_dataset[0]
        x, edge_index, y = data.x, data.edge_index, data.y
        x_list.append(x)
        y_list.append(y)
        edge_index_list.append(edge_index + idx_shift)
        env_list.append(torch.ones(x.size(0)) * i)
        node_idx_list.append(torch.arange(data.num_nodes) + idx_shift)

        idx_shift += data.num_nodes
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    env = torch.cat(env_list, dim=0)
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset.env = env
    dataset.env_num = len(sub_graphs)
    dataset.train_env_num = train_num

    assert (train_num <= 5)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio) : int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [node_idx_list[-1]] if train_num>=4 else node_idx_list[train_num:]

    return dataset

def load_synthetic_dataset(data_dir, name, env_num=6, train_num=3, train_ratio=0.5, valid_ratio=0.25, combine=False):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                            name=name, transform=transform)
    preprocess_dir = os.path.join(data_dir, 'Planetoid', name)

    data = torch_dataset[0]

    edge_index = data.edge_index
    x = data.x
    d = x.shape[1]

    preprocess_dir = os.path.join(preprocess_dir, 'gen')
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    spu_feat_num = 10
    class_num = data.y.max().item() + 1

    node_idx_list = [torch.arange(data.num_nodes) + i*data.num_nodes for i in range(env_num)]

    file_path = preprocess_dir + f'/{class_num}-{spu_feat_num}-{env_num}.pkl'
    if not os.path.exists(file_path):

        print("creating new synthetic data...")
        x_list, edge_index_list, y_list, env_list = [], [], [], []
        idx_shift = 0

        # Generator_y = GCN_gen(in_channels=d, hidden_channels=10, out_channels=class_num, num_layers=2)
        Generator_x = GCN_gen(in_channels=class_num, hidden_channels=10, out_channels=spu_feat_num, num_layers=2)
        Generator_noise = nn.Linear(env_num, spu_feat_num)

        with torch.no_grad():
            for i in range(env_num):
                label_new = F.one_hot(data.y, class_num).squeeze(1).float()
                context_ = torch.zeros(x.size(0), env_num)
                context_[:, i] = 1
                x2 = Generator_x(label_new, edge_index) + Generator_noise(context_)
                x2 += torch.ones_like(x2).normal_(0, 0.1)
                x_new = torch.cat([x, x2], dim=1)

                x_list.append(x_new)
                y_list.append(data.y)
                edge_index_list.append(edge_index + idx_shift)
                env_list.append(torch.ones(x.size(0)) * i)

                idx_shift += data.num_nodes

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        env = torch.cat(env_list, dim=0)
        dataset = Data(x=x, edge_index=edge_index, y=y)
        dataset.env = env

        with open(file_path, 'wb') as f:
            pkl.dump((dataset), f, pkl.HIGHEST_PROTOCOL)
    else:
        print("using existing synthetic data...")
        with open(file_path, 'rb') as f:
            dataset = pkl.load(f)

    assert (train_num <= env_num-1)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio): int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
   
    if combine:
        dataset.test_ood_idx = [node_idx_list[-1]] if train_num==env_num-1 else [torch.cat(node_idx_list[train_num:], dim=0)] # Combine three ood environments
    else:
        dataset.test_ood_idx = [node_idx_list[-1]] if train_num==env_num-1 else node_idx_list[train_num:] # Test three ood environments respectively
    
    dataset.env_num = env_num
    dataset.train_env_num = train_num

    return dataset


def load_arxiv_dataset(data_dir, train_num=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')

    node_years = ogb_dataset.graph['node_year']

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    year_bound = [2005, 2010, 2012, 2014, 2016, 2018, 2021]
    env = torch.zeros(label.shape[0])
    for n in range(node_years.shape[0]):
        year = int(node_years[n])
        for i in range(len(year_bound)-1):
            if year >= year_bound[i+1]:
                continue
            else:
                env[n] = i
                break

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    dataset.env = env
    dataset.env_num = len(year_bound)
    dataset.train_env_num = train_num

    ind_mask = (node_years < year_bound[train_num]).squeeze(1)
    idx = torch.arange(dataset.num_nodes)
    ind_idx = idx[ind_mask]
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    dataset.test_ood_idx = []

    for i in range(train_num, len(year_bound)-1):
        ood_mask_i = ((node_years >= year_bound[i]) * (node_years < year_bound[i+1])).squeeze(1)
        dataset.test_ood_idx.append(idx[ood_mask_i])

    return dataset


def load_elliptic_dataset(data_dir, train_num=5, train_ratio=0.5, valid_ratio=0.25):

    sub_graphs = range(0, 49)
    x_list, edge_index_list, y_list, mask_list, env_list = [], [], [], [], []
    node_idx_list = []
    idx_shift = 0
    for i in sub_graphs:
        result = pkl.load(open('{}/elliptic/{}.pkl'.format(data_dir, i), 'rb'))
        A, label, features = result
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label)

        x_list.append(x)
        y_list.append(y)
        mask = (y >= 0)
        edge_index_list.append(edge_index + idx_shift)
        env_list.append(torch.ones(x.size(0)) * i)
        node_idx_list.append(torch.arange(x.shape[0])[mask] + idx_shift)

        idx_shift += x.shape[0]

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    env = torch.cat(env_list, dim=0)
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset.env = env
    dataset.env_num = len(sub_graphs)
    dataset.train_env_num = train_num

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio): int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    ood_margin = 4
    dataset.test_ood_idx = []
    for k in range((len(sub_graphs) - train_num*2) // ood_margin - 1):
        ood_idx_k = [node_idx_list[l] for l in range(train_num*2 + ood_margin * k, train_num*2 + ood_margin * (k + 1))]
        dataset.test_ood_idx.append(torch.cat(ood_idx_k, dim=0))
    return dataset