'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
'''

# TODO(Tianju): 
# data_Y is not useful, eliminate it
# should use args to store input size 31


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from .. import arguments
from .models import NeuralNetSolver


def loss_function(x_source, x_solution):
    # The loss is defined inspired by minimizing variational energy
    # Only first order derivative of x_solution is needed

    # x_source should be torch array with shape (batch, 1, 31, 31)
    # x_solution should be torch array with shape (batch, 1, 31, 31)
    assert(x_source.shape == x_solution.shape)
    assert(len(x_source.shape) == 4)

    domain_size = x_source.shape[-1]
    batch_size = x_source.shape[0]

    # Prepared for finite diffence
    h = 0.5
    mesh_x = h*torch.ones((batch_size, 1, domain_size, domain_size))
    mesh_x[:, :, :, :1] = 0.5*mesh_x[:, :, :, :1]
    mesh_x[:, :, :, -1:] = 0.5*mesh_x[:, :, :, -1:]
    mesh_y = h*torch.ones((batch_size, 1, domain_size, domain_size))
    mesh_y[:, :, :1, :] = 0.5*mesh_y[:, :, :1, :]
    mesh_y[:, :, -1:, :] = 0.5*mesh_y[:, :, -1:, :]
    area = mesh_x*mesh_y

    left_x = torch.cat( (x_solution[:, :, :, :1], x_solution[:, :, :, :-1]), dim=3)
    right_x = torch.cat( (x_solution[:, :, :, 1:], x_solution[:, :, :, -1:]), dim=3)
    down_x = torch.cat( (x_solution[:, :, 1:, :], x_solution[:, :, -1:, :]), dim=2)
    up_x = torch.cat( (x_solution[:, :, :1, :], x_solution[:, :, :-1, :]), dim=2)

    grad_u_x = (right_x - left_x) / (2*mesh_x)
    grad_u_y = (up_x - down_x) / (2*mesh_y)

    alpha = 1.
    loss_grad = 0.5*(grad_u_x**2 + grad_u_y**2)*area
    loss_source = -x_source*x_solution*area
    # Zero Dirichlet bc for now
    # TODO(Tianju): Change this to a more general bc
    loss_bc = x_solution[:, :, :, :1]**2 * mesh_y[:, :, :, :1] + \
              x_solution[:, :, :, -1:]**2 * mesh_y[:, :, :, -1:] + \
              x_solution[:, :, :1, :]**2 + mesh_x[:, :, :1, :] + \
              x_solution[:, :, -1:, :]**2 + mesh_x[:, :, -1:, :]
    total_loss = loss_grad.sum() + loss_source.sum() + alpha*loss_bc.sum()

    return total_loss

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(data, recon_batch)
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

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.float()
            recon_batch = model(data)
            test_loss += loss_function(data, recon_batch).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def shuffle_data(data_X, data_Y):
    n_samps = len(data_X)
    n_train = int((1 - 0.2) * n_samps)
    inds = np.random.permutation(n_samps)    
    inds_train = inds[:n_train]
    inds_test  = inds[n_train:]
    train_X = data_X[inds_train]
    train_Y = data_Y[inds_train]
    test_X = data_X[inds_test]
    test_Y = data_Y[inds_test]
    return train_X, train_Y, test_X, test_Y

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

if __name__ == "__main__":
    args = arguments.args
 
    torch.manual_seed(args.seed)
    data_X = np.load(args.root_path + '/' + args.numpy_path + '/GP-3000-1-31-31.npy')

    # Everything related to Y is useless, should be elimiated
    data_Y = data_Y = np.zeros((data_X.shape[0], 1)) 
    train_X, train_Y, test_X, test_Y = shuffle_data(data_X, data_Y)
    train_X, \
    train_Y, \
    test_X, \
    test_Y = \
    torch.tensor(train_X).float(), \
    torch.tensor(train_Y).float(), \
    torch.tensor(test_X).float(), \
    torch.tensor(test_Y).float() 


    train_X = impose_bc(train_X)
    test_X = impose_bc(test_X)


    train_data = TensorDataset(train_X, train_Y)
    test_data = TensorDataset(test_X, test_Y)
    train_loader =  DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    analysis_mode = True

    if analysis_mode:
        model_path = args.root_path + '/' + args.model_path + '/model'
        model =  torch.load(model_path)
        # print(model.fc.weight.data.numpy())
        # print(model.fc.bias.data.numpy())
        input_data = test_input()
        input_data = impose_bc(input_data)
        output_data = model(input_data)
        save_solution(output_data, args, "star_solution")
    else:
        model = NeuralNetSolver(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
        torch.save(model, args.root_path + '/' + args.model_path + '/model')