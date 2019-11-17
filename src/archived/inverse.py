'''Inverse problem
'''
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
from .. import arguments
from .models import NeuralNetSolver
from .trainer import test_input, loss_function, save_solution


def index_to_coor(n1, n2):
    x1 = args.h*n2
    x2 = (args.edge - 1 - n1)*args.h
    return x1, x2

def u_func(x1, x2):
    l = args.h*(args.edge - 1)
    return 0.005*x1*(l - x1)*x2*(l - x2)

def generate_u():
    u_target = torch.zeros((args.edge, args.edge), dtype=torch.float) 
    for i, u_t in enumerate(u_target):   
        for j, _ in enumerate(u_t):
            x1, x2 = index_to_coor(i, j)
            u_target[i][j] = u_func(x1, x2)
    return u_target

def regularization(x):
    left_x = torch.cat( (x[:, :, :, :1], x[:, :, :, :-1]), dim=3)
    right_x = torch.cat( (x[:, :, :, 1:], x[:, :, :, -1:]), dim=3)
    down_x = torch.cat( (x[:, :, 1:, :], x[:, :, -1:, :]), dim=2)
    up_x = torch.cat( (x[:, :, :1, :], x[:, :, :-1, :]), dim=2)

    penalty = 1*(right_x + left_x + down_x + up_x - 4*x)**2

    return penalty.sum()   

def objective(x):
    x = x.reshape(1, 1, args.edge, args.edge)
    x = torch.tensor(x, requires_grad=False, dtype=torch.float)
    output_u = network(x)
    target_u = u.view(1, 1, args.edge, args.edge)
    l2_error = torch.norm(output_u - target_u, 2).data.numpy()
    reg_error = regularization(x).data.numpy()
    print(l2_error)
    print(reg_error)
    return l2_error + reg_error

def objective_der(x):
    x = x.reshape(1, 1, args.edge, args.edge)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float)
    output_u = network(x)
    target_u = u.view(1, 1, args.edge, args.edge)
    l2_error = torch.norm(output_u - target_u, 2)
    reg_error = regularization(x)
    error = l2_error + reg_error
    error.backward(retain_graph=True)
    grad = x.grad
    grad = grad.data.numpy().flatten().astype(np.float64)
    return grad

def ground_truth(x1, x2):
    l = args.h*(args.edge - 1)
    return 0.005*2*(x1*(l - x1) + x2*(l - x2))

def generate_f():
    f_target = torch.zeros((args.edge, args.edge), dtype=torch.float) 
    for i, f_t in enumerate(f_target):   
        for j, _ in enumerate(f_t):
            x1, x2 = index_to_coor(i, j)
            f_target[i][j] = ground_truth(x1, x2)
    return f_target


if __name__ == "__main__":
    args = arguments.args
    model_path =args.root_path + '/' + args.model_path + '/model'
    network =  torch.load(model_path)

    input_data = torch.ones((1, 1, args.edge, args.edge), dtype=torch.float)
    u = network(input_data)
    u = generate_u()
    f = generate_f()
    f_np = f.data.numpy()

    x_initial = 0*np.ones_like(u.data.numpy().flatten())

    output = network(f.view(1, 1, args.edge, args.edge))

    x_initial = f_np.flatten()
    options={'xtol': 1e-15, 'eps': 1e-15, 'maxiter': 1000, 'disp': True, 'return_all': True}
    res = opt.minimize(objective,
                       x_initial, 
                       method='L-BFGS-B', 
                       jac=objective_der,
                       callback=None,
                       options=None)

    # L-BFGS-B > CG > BFGS > Newton-CG
    
    # print(res.x.reshape(args.edge, args.edge))

    print("min of optimized", np.min(res.x.flatten()))
    print("max of optimized",np.max(res.x.flatten()))

    print("min of ground_truth", np.min(f_np.flatten()))
    print("max of ground_truth", np.max(f_np.flatten()))
 
    save_solution(torch.tensor(res.x), args, "optimized")
    save_solution(f, args, "ground_truth")

    # best_x = torch.tensor(res.x, dtype=torch.float).view(1, 1,  args.edge, args.edge)
    # save_solution(network(best_x), args, "trouble")
    # save_solution(output, args, "optimized")
    # save_solution(u, args, "ground_truth")
    # l2_error = torch.norm(output - u, 2)
    # print(l2_error)
    
    exit()

