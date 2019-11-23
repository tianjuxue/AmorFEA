'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Linear case
'''

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


def batch_mm(matrix, vector_batch):
    batch_size = vector_batch.shape[0]
    # Stack the vector batch into columns. (b, n, 1) -> (n, b)
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, b) -> (b, m, 1)
    return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1, 1)

def loss_function(x_source, x_solution, A_sp):
    # loss function is defined so that PDE is satisfied

    # x_source should be torch tensor with shape (batch, input_size)
    # x_solution should be torch tensor with shape (batch, input_size)
    assert(x_source.shape == x_solution.shape and len(x_source.shape) == 2)
 
    # to (batch, input_size, 1) for batch matrix-vector multiplication
    x_solution = x_solution.unsqueeze(2)

    # Sparse representation is ~10 times faster then dense representation in this case
    # dense representation would be lhs = torch.matmul(A, x_solution)
    lhs = batch_mm(A_sp, x_solution)

    assert(len(lhs.shape) == 3 and lhs.shape[2] == 1) 

    lhs = lhs.squeeze()
    rhs = x_source
    loss = ((lhs - rhs)**2).sum()
    return loss

def train(epoch, A_sp):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        # In case we need second order optimizer, a closure has to be defined
        def closure():
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = loss_function(data, recon_batch, A_sp)
            loss.backward()
            return loss

        data = data[0].float()
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(data, recon_batch, A_sp)
        loss.backward()
        train_loss += loss.item()
        optimizer.step(closure)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
    return train_loss

def test(epoch, A_sp):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data[0].float()
            recon_batch = model(data)
            test_loss += loss_function(data, recon_batch, A_sp).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.6f}'.format(test_loss))
    W_trained = model.fc.weight.data.numpy()

    L_inf = np.max(np.abs(W_trained - W_true))
    L_fro = np.linalg.norm(np.abs(W_trained - W_true))
    print('====> L_inf norm for matrix diff is {}'.format(L_inf))
    print('====> L_fro norm for matrix diff is {}'.format(L_fro))
    print('\n\n')
    return test_loss, L_inf, L_fro

def shuffle_data(data_X):
    n_samps = len(data_X)
    n_train = int((1 - 0.1) * n_samps)
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


def ground_truth(graph):
    gradient_x1_operator = torch.tensor(graph.gradient_x1).float()
    gradient_x2_operator = torch.tensor(graph.gradient_x2).float()
    boundary_operator = torch.tensor(graph.reset_matrix_boundary).float()
    interior_operator = torch.tensor(graph.reset_matrix_interior).float()
    operators = [gradient_x1_operator, gradient_x2_operator, boundary_operator, interior_operator]

    tmp = -torch.matmul(gradient_x1_operator, gradient_x1_operator) \
           -torch.matmul(gradient_x2_operator, gradient_x2_operator)
    tmp /= graph.num_vertices
    A = torch.matmul(interior_operator, tmp) + boundary_operator

    # If A is ill-conditioned, convergence will be painfully slow
    # print(np.max(A.data.numpy()), np.min(A.data.numpy()))

    A_inv = A.inverse()
    A_sp = A.to_sparse()

    return A, A_sp, A_inv

def save_progress(np_data, model, L_inf, L_fro, milestone, counter):
    if L_inf < milestone[counter]:
        torch.save(model, args.root_path + '/' + args.model_path + '/model_' + str(counter))
        np.save(args.root_path + '/' + args.numpy_path + '/error_' + str(counter), np_data)
        counter += 1
    return counter

if __name__ == "__main__":
    args = arguments.args
    torch.manual_seed(args.seed)
    graph = GraphManual(args)
    # graph = GraphMSHR(args)
    args.input_size = graph.num_vertices
    A, A_sp, A_inv = ground_truth(graph)
    W_true = A_inv.data.numpy()
    
    data_X = np.load(args.root_path + '/' + args.numpy_path + '/' + graph.name +
                     '-Uniform-30000-' + str(graph.num_vertices) + '.npy')
    # data_X = np.load(args.root_path + '/' + args.numpy_path + '/' + graph.name +
    #                  '-Multi-30000-' + str(graph.num_vertices) + '.npy')

 
    train_X, test_X = shuffle_data(data_X)
    train_X = torch.tensor(train_X).float().view(train_X.shape[0], args.input_size)
    test_X = torch.tensor(test_X).float().view(test_X.shape[0], args.input_size)

    train_data = TensorDataset(train_X)
    test_data = TensorDataset(test_X)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    analysis_mode = False

    # np_data = np.load(args.root_path + '/' + args.numpy_path + '/error_1.npy')
    # print(np_data)
    # exit()

    if analysis_mode:
        model_path = args.root_path + '/' + args.model_path + '/model'
        model =  torch.load(model_path)
        # print(model.fc.bias.data.numpy())
        W_trained = model.fc.weight.data.numpy()
        print(W_trained)
        print('\n\n\n')
        print(W_true.data.numpy())
        print(np.max(np.abs(W_trained - W_true)))

        # input_data = test_input()
        # input_data = impose_bc(input_data)
        # output_data = model(input_data)
        # save_solution(output_data, args, "star_solution")
    else:
        model = NeuralNetSolver(args)
        # model_path = args.root_path + '/' + args.model_path + '/model_tmp'
        # model =  torch.load(model_path)    
        model.fc.weight.data = torch.zeros((args.input_size, args.input_size))

        # optimal tuning lr=1e-1, momentum=0.6 
        # optimizer = optim.Adam(model.parameters(), lr=1e-2)
        # 11 steps down to error of 1
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.85)
        # optimizer = optim.LBFGS(model.parameters(), lr=1e-2, max_iter=20, history_size=40)

        milestone = [ 2**(-i) for i in range(-1, 11) ]
        counter = 0
        np_data = np.zeros((2, args.epochs))
        for epoch in range(args.epochs):
            train_loss = train(epoch, A_sp)
            test_loss, L_inf, L_fro = test(epoch, A_sp)
            np_data[:, epoch] = [L_inf, L_fro]
            counter = save_progress(np_data, model, L_inf, L_fro, milestone, counter)
            
            if L_inf < 1e-4:
                break

        torch.save(model, args.root_path + '/' + args.model_path + '/model')