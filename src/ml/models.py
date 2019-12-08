import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .trainer import batch_mat_mat, batch_mat_vec, batch_mat_vec_2


class LinearRegressor(nn.Module):
    def __init__(self, args, graph_info):
        super(LinearRegressor, self).__init__()
        self.args = args
        self.bc_value, self.interior_flag, _, self.B_sp = graph_info
        self.fc = nn.Linear(args.input_size, args.input_size, bias=False)

    def forward(self, x):
        x = batch_mat_vec(self.B_sp, x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        x = self.fc(x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        return x


class MLP(nn.Module):
    def __init__(self, args, graph_info):
        super(MLP, self).__init__()
        self.args = args
        self.bc_value, self.interior_flag, _, self.B_sp = graph_info
        self.fcc = nn.Sequential(nn.Linear(args.input_size, args.input_size),
                                 nn.SELU(), 
                                 nn.Linear(args.input_size, args.input_size),
                                 nn.SELU(), 
                                 nn.Linear(args.input_size, args.input_size))

    def forward(self, x):
        x = batch_mat_vec(self.B_sp, x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        x = self.fcc(x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)

        output = batch_mat_mat(adj, support)
        # output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MixedNetwork(nn.Module):
    def __init__(self, args, graph_info):
        super(MixedNetwork, self).__init__()
        self.args = args
        self.bc_value, self.interior_flag, self.adj, self.B_sp = graph_info

        self.gc1 = GraphConvolution(1, 20)
        self.gc2 = GraphConvolution(20, 1)

        self.fcc = nn.Sequential(nn.Linear(args.input_size, args.input_size),
                                 nn.SELU(), 
                                 nn.Linear(args.input_size, args.input_size))

    def forward(self, x):
        x = batch_mat_vec(self.B_sp, x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)

        x = x.unsqueeze(2)
        x = F.selu(self.gc1(x, self.adj))
        x = F.selu(self.gc2(x, self.adj))
        x = x.squeeze()

        x = self.fcc(x)  
        x = torch.addcmul(self.bc_value, x, self.interior_flag)

        return x


class GCN(nn.Module):
    def __init__(self, args, graph_info):
        super(GCN, self).__init__()
        self.args = args
        self.bc_value, self.interior_flag, self.adj, self.B_sp = graph_info
        self.gc1 = GraphConvolution(1, 20)
        self.gc2 = GraphConvolution(20, 1)

    def forward(self, x):
        x = batch_mat_vec(self.B_sp, x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)

        x = x.unsqueeze(2)
        x = F.selu(self.gc1(x, self.adj))
        x = F.selu(self.gc2(x, self.adj))
        x = x.squeeze()

        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        return x