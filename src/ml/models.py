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
        self.mat_list, self.joints, self.coo_diff, self.shapes = graph_info
        self.fcc1 = nn.Sequential(nn.Linear(args.input_size, args.input_size, bias=True),
                                  nn.Sigmoid())

        self.fcc2 = nn.Sequential(nn.Linear(self.shapes[0], self.shapes[1], bias=True))
                                                  
        self.reset_parameters()

    def reset_parameters(self):
        #TODO(Tianju): We don't need distributions here
        for i, layer in enumerate(self.fcc1):
            if i%2 == 0:
                layer.weight.data[:] = 0
                layer.bias.data[:] = 0

        for i, layer in enumerate(self.fcc2):
            if i%2 == 0:
                layer.weight.data[:] = 0
                layer.bias.data[:] = 0

    def forward(self, x):
        angles = 0.5*np.pi + 0.5*np.pi*2*(self.fcc1(x) - 0.5)
        lx_u, ly_u, rx_u, ry_u, bc_u = constrain(x, self.mat_list, self.coo_diff, 
                                                 self.joints, angles)
        int_u = self.fcc2(bc_u)
        int_u = batch_mat_vec(self.mat_list[-1].transpose(0, 1), int_u)
        u = lx_u + ly_u + rx_u + ry_u + int_u
        return u


class RobotSolver(nn.Module):
    def __init__(self, args, graph_info):
        super(RobotSolver, self).__init__()
        self.args = args
        self.mat_list, self.joints, self.coo_diff, self.shapes = graph_info
        self.para_angles = Parameter(torch.zeros(self.shapes[0]//2) + 0.5*np.pi )
        self.para_disp = Parameter(torch.zeros(self.shapes[1]))

    def forward(self, x):
        lx_u, ly_u, rx_u, ry_u, bc_u = constrain(x, self.mat_list, self.coo_diff, 
                                                 self.joints, self.para_angles.unsqueeze(0))
        int_u = batch_mat_vec(self.mat_list[-1].transpose(0, 1), self.para_disp.unsqueeze(0))
        u = lx_u + ly_u + rx_u + ry_u + int_u
        return u


'''Commonly used functions'''
def constrain(x, mat_list, coo_diff, joints, angles): 
    half_size = x.shape[1]//2
    ratio = 1 + (torch.sigmoid(x) - 0.5)

    new_rods_left = ratio[:, :half_size] * joints[0]
    new_rods_right = ratio[:, half_size:] * joints[1]

    new_rods_left = new_rods_left.unsqueeze(1).repeat(1, new_rods_left.shape[1], 1).triu()
    new_rods_right = new_rods_right.unsqueeze(1).repeat(1, new_rods_right.shape[1], 1).triu()

    cos_angle_left = torch.cos(angles[:, :half_size]).unsqueeze(2)
    sin_angle_left = torch.sin(angles[:, :half_size]).unsqueeze(2)
    cos_angle_right = torch.cos(angles[:, half_size:]).unsqueeze(2)
    sin_angle_right = torch.sin(angles[:, half_size:]).unsqueeze(2)

    lx_u = torch.matmul(new_rods_left, cos_angle_left).squeeze(2) + coo_diff[0]
    ly_u = torch.matmul(new_rods_left, sin_angle_left).squeeze(2) + coo_diff[1]
    rx_u = torch.matmul(new_rods_right, cos_angle_right).squeeze(2) + coo_diff[2]
    ry_u =  torch.matmul(new_rods_right, sin_angle_right).squeeze(2) + coo_diff[3]
    bc_u = torch.cat([lx_u, ly_u, rx_u, ry_u], dim=1)

    lx_u = batch_mat_vec(mat_list[1].transpose(0, 1), lx_u)
    ly_u = batch_mat_vec(mat_list[2].transpose(0, 1), ly_u)
    rx_u = batch_mat_vec(mat_list[3].transpose(0, 1), rx_u)
    ry_u = batch_mat_vec(mat_list[4].transpose(0, 1), ry_u)

    return lx_u, ly_u, rx_u, ry_u, bc_u