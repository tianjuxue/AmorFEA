#TODO(Tianju)
# Change the fully connected network to a more advanced architecture like CNN

from torch import nn
from torch.nn import functional as F
import torch


class MLP(nn.Module):
    def __init__(self, args, graph_info):
        super(MLP, self).__init__()
        self.args = args
        self.bc_value, self.interior_flag = graph_info
        self.encoder = nn.Sequential(
            nn.Linear(args.input_size, args.input_size),
            nn.SELU(True),
            nn.Linear(args.input_size, args.input_size),
            nn.SELU(True))
        self.decoder = nn.Sequential(
            nn.Linear(args.input_size, args.input_size),
            nn.SELU(True),
            nn.Linear(args.input_size, args.input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.addcmul(self.bc_value, x, self.interior_flag)
        return x

class LinearRegressor(nn.Module):
    def __init__(self, args):
        super(LinearRegressor, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.input_size, args.input_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x

# class GraphNN(nn.Module):
#     def __init__(self, args, operators):
#         super(GraphNN, self).__init__()
#         self.args = args
#         self.operators = operators
#         self.fc = nn.Linear(args.input_size, args.input_size, bias=False)

#     def forward(self, x):
#         x = self.fc(x)


#         torch.addcmul(bc_value, value=1, input_x, interior_flag, out=None)


#         return x