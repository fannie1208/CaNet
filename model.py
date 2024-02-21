import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class CaNetConv(nn.Module):

    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CaNetConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, e, weights=None):
        if weights == None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D]
            h = torch.matmul(xi, weights) # [K, N, D]
            N = x.size()[0]
            adj, _ = remove_self_loops(adj)
            adj, _ = add_self_loops(adj, num_nodes=N)
            edge_h = torch.cat((h[:, adj[0, :], :], h[:, adj[1, :], :]), dim=2)  # [K, E, 2*D]
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max , _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits-logits_max)  # [K, E]

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E]
                e_expsum_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), torch.ones(N, 1).cuda()) + eps
                assert not torch.isnan(e_expsum_k).any()

                hi_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1) # [N, K, D]

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)  # [N, K, D]
        output = torch.sum(torch.mul(es, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output

class CaNet(nn.Module):
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.env_type == 'node':
                self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            else:
                raise NotImplementedError
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.env_type = args.env_type
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.env_enc:
            enc.reset_parameters()

    def forward(self, x, adj, idx=None, training=False):
        self.training = training
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                if self.env_type == 'node':
                    logit = self.env_enc[i](h)
                else:
                    logit = self.env_enc[i](h, adj, h0)
                e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                reg += self.reg_loss(e, logit)
            else:
                if self.env_type == 'node':
                    e = F.softmax(self.env_enc[i](h), dim=-1)
                else:
                    e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
            h = self.act_fn(con(h, adj, e))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(
            torch.mul(z, log_pi), dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def loss_compute(self, d, criterion, args):
        logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        loss = sup_loss + args.lamda * reg_loss
        return loss