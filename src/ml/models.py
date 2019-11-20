#TODO(Tianju)
# Change the fully connected network to a more advanced architecture like CNN

from torch import nn, optim
from torch.nn import functional as F
import torch


# class NeuralNetSolver(nn.Module):
#     def __init__(self, args):
#         super(NeuralNetSolver, self).__init__()
#         self.args = args
#         self.encoder = nn.Sequential(
#             nn.Linear(args.input_size, args.input_size),
#             nn.SELU(True))
#         self.decoder = nn.Sequential(
#             nn.Linear(args.input_size, args.input_size),
#             nn.SELU(True))

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class NeuralNetSolver(nn.Module):
    def __init__(self, args):
        super(NeuralNetSolver, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.input_size, args.input_size, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x

# class ConvSolver(nn.Module):
#     def __init__(self, args):
#         super(ConvSolver, self).__init__()
#         self.args = args
#         # O = (I - K + 2P)/S + 1
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=1, padding=1),   
#             nn.ReLU(True),
#             nn.MaxPool2d(3, stride=1, padding=1)   
#         )

#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         x = self.decode(z)
#         y = self.predict(z)
#         return x, y, mu, logvar