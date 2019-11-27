import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from .. import arguments
from .models import NeuralNetSolver
from .linear import test_input, loss_function, save_solution, ground_truth
from ..graph.domain import GraphManual, GraphMSHR
import scipy.optimize as opt


def objective_der(x):
    der = 2*np.matmul(np.matmul(A_T, A), x) - 2*a
    return der

def objective(x):
    tmp = np.matmul(A , x)  
    obj = np.inner(tmp, tmp) - 2*np.inner(a, x) + 1
    return obj

def call_back(x):
    print(objective(x))

if __name__ == "__main__":
    args = arguments.args
    graph = GraphManual(args)
    M = graph.num_vertices
    col = 8
    gradient_x1_operator = graph.gradient_x1
    gradient_x2_operator = graph.gradient_x2
    boundary_operator = graph.reset_matrix_boundary
    interior_operator = graph.reset_matrix_interior
    A = -np.matmul(gradient_x2_operator, gradient_x2_operator) \
        -np.matmul(gradient_x1_operator, gradient_x1_operator)
    A /= M
    A = np.matmul(interior_operator, A) + boundary_operator
    A_T = np.transpose(A)
    a = A[col,:]
    A_inv = np.linalg.inv(A)

    print(np.max(A), np.min(A))
    print(np.max(A_inv), np.min(A_inv))
    # exit()

    x_initial = np.zeros(M)

    options={'eps': 1e-10, 'maxiter': 1000, 'disp': True, 'return_all': True}

    res = opt.minimize(objective,
                       x_initial, 
                       method='CG', 
                       jac=objective_der,
                       callback=call_back,
                       options=options)

    # print(np.max(A))
    # print(np.max(A_inv))
    print(A_inv[:,col])
    print('\n\n')
    print(res.x)

    # CG > BFGS > Newton-CG
