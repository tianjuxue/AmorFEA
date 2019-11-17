'''Use the neural network as a PDE solver
Input is fixed to represent a certain source term
'''

import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from .. import arguments
from .models import NeuralNetSolver
from .trainer import test_input, loss_function, save_solution


if __name__ == "__main__":
    args = arguments.args

    input_data = torch.ones((1, 1, 31, 31), dtype=torch.float)
    model = NeuralNetSolver(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output_data = model(input_data)
        loss = loss_function(input_data, output_data)
        loss.backward()
        optimizer.step()
        print(loss.item())

    save_solution(output_data, args)
    torch.save(model, args.root_path + '/' + args.model_path + '/model')