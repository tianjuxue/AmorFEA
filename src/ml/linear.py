'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
'''

# TODO(Tianju): 
# data_Y is not useful, eliminate it
# should use args to store input size 31


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from .. import arguments
from .models import NeuralNetSolver
from ..graph.domain import GraphManual, GraphMSHR


def loss_function(x_source, 
                  x_solution, 
                  operators):
    # loss function is defined so that PDE is satisfied

    gradient_x1_operator, \
    gradient_x2_operator, \
    boundary_operator, \
    interior_operator \
    = operators

    # x_source should be torch tensor with shape (batch, input_size)
    # x_solution should be torch tensor with shape (batch, input_size)
    assert(x_source.shape == x_solution.shape and len(x_source.shape) == 2)
 
    # to (batch, input_size, 1) for batch matrix-vector multiplication
    x_solution = x_solution.unsqueeze(2)

    # negative laplacian
    neg_lap = -torch.matmul(gradient_x1_operator, torch.matmul(gradient_x1_operator, x_solution)) \
              -torch.matmul(gradient_x2_operator, torch.matmul(gradient_x2_operator, x_solution))

    lhs = torch.matmul(interior_operator, neg_lap) + torch.matmul(boundary_operator, x_solution)
    assert(len(lhs.shape) == 3 and lhs.shape[2] == 1) 
    lhs = lhs.squeeze()
    rhs = x_source

    loss = ((lhs - rhs)**2).sum()
    return loss

def train(epoch, operators):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].float()
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(data, recon_batch, operators)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch, operators):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data[0].float()
            recon_batch = model(data)
            test_loss += loss_function(data, recon_batch, operators).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def shuffle_data(data_X):
    n_samps = len(data_X)
    n_train = int((1 - 0.2) * n_samps)
    inds = np.random.permutation(n_samps)    
    inds_train = inds[:n_train]
    inds_test  = inds[n_train:]
    train_X = data_X[inds_train]
    test_X = data_X[inds_test]
    return train_X, test_X

def test_input():
    h = 0.5
    N = 30
    L = N*h
    input_data = torch.ones((31, 31), dtype=torch.float)
    for i, row in enumerate(input_data):
        for j, _ in enumerate(row):
            x_coor = j*h
            input_data[i][j] = np.sin(2*np.pi/L*x_coor)

    input_data = torch.ones((1, 1, 31, 31), dtype=torch.float)
    
    return input_data.view(1, 1, 31, 31)

def save_solution(output_data, args, name):
    solution_np = output_data.data.numpy().reshape(31, 31)
    plt.imshow(solution_np, cmap='bwr')  
    plt.axis('off') 
    plt.savefig(args.root_path + '/' + args.images_path + "/" + name + ".png", bbox_inches='tight')
    print(output_data.max())

def impose_bc(x):
    x[:, :, :, :1] = 0
    x[:, :, :, -1:] = 0      
    x[:, :, :1, :] = 0
    x[:, :, -1:, :] = 0
    return x

def ground_truth(operators):
    tmp = -torch.matmul(gradient_x1_operator, gradient_x1_operator) \
           -torch.matmul(gradient_x2_operator, gradient_x2_operator)
    A = torch.matmul(interior_operator, tmp) + boundary_operator
    # print(A.data.numpy())
    A_inv = A.inverse()
    return A_inv.data.numpy()



if __name__ == "__main__":
    args = arguments.args
    graph = GraphManual(args)
    # graph = GraphMSHR(args)
    args.input_size = graph.num_vertices

    gradient_x1_operator = torch.tensor(graph.gradient_x1).float()
    gradient_x2_operator = torch.tensor(graph.gradient_x2).float()
    boundary_operator = torch.tensor(graph.reset_matrix_boundary).float()
    interior_operator = torch.tensor(graph.reset_matrix_interior).float()
    operators = [gradient_x1_operator, gradient_x2_operator, boundary_operator, interior_operator]

    torch.manual_seed(args.seed)
    data_X = np.load(args.root_path + '/' + args.numpy_path + '/' + graph.name +
                     '-GP-300000-' + str(graph.num_vertices) + '.npy')

    # Everything related to Y is useless, should be elimiated
    train_X, test_X = shuffle_data(data_X)
    train_X = torch.tensor(train_X).float().view(train_X.shape[0], args.input_size)
    test_X = torch.tensor(test_X).float().view(test_X.shape[0], args.input_size)


    # train_X = impose_bc(train_X)
    # test_X = impose_bc(test_X)

    train_data = TensorDataset(train_X)
    test_data = TensorDataset(test_X)
    train_loader =  DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    analysis_mode = False

    if analysis_mode:
        model_path = args.root_path + '/' + args.model_path + '/model'
        model =  torch.load(model_path)
        # print(model.fc.bias.data.numpy())
        W_trained = model.fc.weight.data.numpy()
        print(W_trained)
        print('\n\n\n')
        W_true = ground_truth(operators)
        print(W_true)
        # print(np.max(np.abs(W_trained - W_true)))

        # input_data = test_input()
        # input_data = impose_bc(input_data)
        # output_data = model(input_data)
        # save_solution(output_data, args, "star_solution")
    else:
        model = NeuralNetSolver(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train(epoch, operators)
            test(epoch, operators)
        torch.save(model, args.root_path + '/' + args.model_path + '/model')