import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .trainer import batch_mat_mat, batch_mat_vec


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


class RobotNetwork(nn.Module):
    def __init__(self, args, graph_info):
        super(RobotNetwork, self).__init__()
        self.args = args
        self.bc_mat_list, self.bc_list, self.joints, self.coo_diff = graph_info
        output_size = len(self.bc_list[0])

        self.interior_flag = torch.ones(len(self.bc_list[0]))
        for bc in self.bc_list:
            self.interior_flag -= bc

        self.fcc1 = nn.Sequential(nn.Linear(args.input_size, args.input_size),
                                  nn.Sigmoid(), 
                                  nn.Linear(args.input_size, args.input_size),
                                  nn.Sigmoid())

        self.fcc2 = nn.Sequential(nn.Linear(args.input_size, output_size),
                                  nn.SELU(), 
                                  nn.Linear(output_size, output_size))        

        self.reset_parameters()

    def reset_parameters(self):
        self.fcc1[0].weight.data.uniform_(-0.001, 0.001)
        self.fcc1[2].weight.data.uniform_(-0.001, 0.001)
        self.fcc2[0].weight.data.uniform_(-0.001, 0.001)
        self.fcc2[2].weight.data.uniform_(-0.001, 0.001)
        self.fcc1[0].bias.data.uniform_(-0.001, 0.001)
        self.fcc1[2].bias.data.uniform_(-0.001, 0.001)
        self.fcc2[0].bias.data.uniform_(-0.001, 0.001)
        self.fcc2[2].bias.data.uniform_(-0.001, 0.001)    
 
    def constrain(self, x):
        half_size = x.shape[1]//2
        angles = 0.5*np.pi + 0.2*np.pi*2*(self.fcc1(x) - 0.5)
        ratio = 1 + (torch.sigmoid(x) - 0.5)

        new_rods_left = ratio[:, :half_size] * self.joints[0]
        new_rods_right = ratio[:, half_size:] * self.joints[1]

        new_rods_left = new_rods_left.unsqueeze(1).repeat(1, new_rods_left.shape[1], 1).tril()
        new_rods_right = new_rods_right.unsqueeze(1).repeat(1, new_rods_right.shape[1], 1).tril()

        cos_angle_left = torch.cos(angles[:, :half_size]).unsqueeze(2)
        sin_angle_left = torch.sin(angles[:, :half_size]).unsqueeze(2)
        cos_angle_right = torch.cos(angles[:, half_size:]).unsqueeze(2)
        sin_angle_right = torch.sin(angles[:, half_size:]).unsqueeze(2)
 
        # print(cos_angle_left[0])
        # print(sin_angle_left[0])
        # print(cos_angle_right[0])
        # print(sin_angle_right[0])
        # # tmp1 = torch.matmul(new_rods_left, cos_angle_left).squeeze(2) + self.coo_diff[0]
        # # tmp2 = torch.matmul(new_rods_left, sin_angle_left).squeeze(2) + self.coo_diff[1]
        # exit()

        lx_u = batch_mat_vec(self.bc_mat_list[1].transpose(0, 1), 
                             torch.matmul(new_rods_left, cos_angle_left).squeeze(2) + self.coo_diff[0])
        ly_u = batch_mat_vec(self.bc_mat_list[2].transpose(0, 1),
                             torch.matmul(new_rods_left, sin_angle_left).squeeze(2) + self.coo_diff[1])
        rx_u = batch_mat_vec(self.bc_mat_list[3].transpose(0, 1),
                             torch.matmul(new_rods_right, cos_angle_right).squeeze(2) + self.coo_diff[2])
        ry_u = batch_mat_vec(self.bc_mat_list[4].transpose(0, 1),
                             torch.matmul(new_rods_right, sin_angle_right).squeeze(2) + self.coo_diff[3])

        # test = torch.sqrt(tmp1[0][0]**2 + tmp2[0][0]**2)
        # print(test.data.numpy())

        return lx_u, ly_u, rx_u, ry_u


    def forward(self, x):
        lx_u, ly_u, rx_u, ry_u = self.constrain(x)
        u = self.fcc2(x)
        bc_values = 0
        u = u * self.interior_flag
        u += lx_u + ly_u + rx_u + ry_u + bc_values
        return u