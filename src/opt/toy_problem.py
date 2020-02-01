import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .. import arguments


class Solver(nn.Module):
    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        self.x = Parameter(torch.zeros(1, 1))
 
    def forward(self, y):
        return self.x

def loss_function(y, x):
    return 0.5*x**2 - x*y

def objective(y, x):
    return (x - 1)**2 + y**2

def forward_prediction_nn(args, source):
    source = np.expand_dims(source, axis=0)
    source = torch.tensor(source, requires_grad=True, dtype=torch.float)
    optimizer = optim.LBFGS(solver.parameters(), lr=1e-1, max_iter=20, history_size=100)
    max_epoch = 10
    tol = 1e-10
    loss_pre, loss_crt = 1, 1
    for epoch in range(max_epoch):
        def closure():
            optimizer.zero_grad()
            solution = solver(source)
            loss = loss_function(source, solution)
            loss.backward(create_graph=True, retain_graph=True)
            return loss

        optimizer.zero_grad()
        solution = solver(source)
        optimizer.step(closure)

        J = torch.autograd.grad(objective(source, solution), source, create_graph=True, retain_graph=True)[0]
        J = J.detach().numpy().flatten()
        print("J is", J)
        print("Optimization for ground truth, loss is", loss_function(source, solution))

    return solution[0].data.numpy(), 

 
def manual_gd():
    y = torch.tensor([2.], requires_grad=True, dtype=torch.float)
    x0 = torch.tensor([0.], requires_grad=True, dtype=torch.float)
    L0 = loss_function(y, x0)
    J0 = torch.autograd.grad(L0, x0, create_graph=True, retain_graph=True)[0]

    for i in range(3):
        x0 = x0 - 0.5*J0
        L0 = loss_function(y, x0)
        J0 = torch.autograd.grad(L0, x0, create_graph=True, retain_graph=True)[0]

    J = torch.autograd.grad(x0, y, create_graph=True, retain_graph=True)[0]
    print(J)

def hessian():
    x = torch.tensor([2., 2.], requires_grad=True, dtype=torch.float)
    y = (x**2).sum()
    J = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]

    print(J)

    H = torch.autograd.grad(J, x, create_graph=True, retain_graph=True)[0]
    print(H)
     

if __name__ == '__main__':
    args = arguments.args
    y = 3
    # x = forward_prediction(args, y)
    # print(x)

    hessian()
