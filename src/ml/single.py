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
from .linear import test_input, loss_function, save_solution, ground_truth, sparse_representation
from ..graph.domain import GraphManual, GraphMSHR


if __name__ == "__main__":
    args = arguments.args
    graph = GraphManual(args)
    args.input_size = graph.num_vertices

    input_data = np.ones(graph.num_vertices)

    A, A_sp, A_inv = ground_truth(graph)
    W_true = A_inv.data.numpy()

    f = np.ones(graph.num_vertices)
    b = np.zeros(graph.num_vertices)
    input_data = np.matmul(graph.reset_matrix_boundary, b) + np.matmul(graph.reset_matrix_interior, f)

    input_data = b
    col = 0
    input_data[col] = 1

    input_data = torch.tensor(input_data, dtype=torch.float)
    input_data = input_data.unsqueeze(0)

    model = NeuralNetSolver(args)

    # Initialize weight to be all zero for faster convergence
    model.fc.weight.data = torch.zeros((args.input_size, args.input_size))

    # optimizer = optim.Adam(model.parameters(), lr=1e-1)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    # optimizer = optim.LBFGS(model.parameters(), lr=1e-1) 

    def closure():
        optimizer.zero_grad()
        output_data = model(input_data)
        loss = loss_function(input_data, output_data, A_sp)
        loss.backward()
        return loss

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output_data = model(input_data)
        loss = loss_function(input_data, output_data, A_sp)
        loss.backward()
        optimizer.step(closure)
        if epoch % args.log_interval == 0:
            print( '====> Loss {}'.format( loss.item() ) )
            W_trained = model.fc.weight.data.numpy()
            print( '====> L_inf norm for column diff {}'.format( np.max( np.abs( (W_trained - W_true)[:, col] ) ) ) )
            print('\n')

    print(input_data.data.numpy())
    print("\n\n\n")
    print(model(input_data).data.numpy())

    # save_solution(output_data, args)
    # torch.save(model, args.root_path + '/' + args.model_path + '/model')