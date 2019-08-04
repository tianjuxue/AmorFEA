#TODO(Tianju)
# Change the fully connected network to a more advanced architecture like CNN

from torch import nn, optim
from torch.nn import functional as F
import torch


class NeuralNetSolver(nn.Module):
    def __init__(self, args):
        super(NeuralNetSolver, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(31 * 31, 128),
            nn.SELU(True),
            nn.Linear(128, 128),
            nn.SELU(True), 
            nn.Linear(128, 128), 
            nn.SELU(True), 
            nn.Linear(128, 128))
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.Linear(128, 128),
            nn.SELU(True), 
            nn.Linear(128, 31 * 31))

    def forward(self, x):
        x = self.encoder(x.view(x.shape[0], 31*31))
        x = self.decoder(x)
        return x.view(x.shape[0], 1, 31, 31)

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