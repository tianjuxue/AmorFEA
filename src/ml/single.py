import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from .. import arguments
from .models import ConvSolver


def loss_function(x_source, x_solution):
    # x_source is torch array with shape (batch, 1, 31, 31)
    # x_solution is torch array with shape (batch, 1, 31, 31)

    assert(x_source.shape == x_solution.shape)
    assert(len(x_source.shape) == 4)

    domain_size = x_source.shape[-1]
    batch_size = x_source.shape[0]

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

if __name__ == "__main__":

    args = arguments.args

    input_data = torch.ones((1, 1, 31, 31), dtype=torch.float)
    model = ConvSolver(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output_data = model(input_data)
        loss = loss_function(input_data, output_data)
        loss.backward()
        optimizer.step()
        print(loss.item())

        solution_np = output_data.data.numpy().reshape(31, 31)
        # fig = plt.figure(epoch)
        plt.imshow(solution_np, cmap='bwr')  
        plt.axis('off') 
        plt.savefig(args.root_path + '/' + args.images_path + "/solution" + str(epoch) + ".png", bbox_inches='tight')

    print(output_data.max())

    torch.save(model, args.root_path + '/' + args.model_path + '/model')
