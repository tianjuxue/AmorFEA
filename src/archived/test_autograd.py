import numpy as np
import torch
from torch import optim


# For given y, minimize this to get x; you get x=y
def loss_function(y, x):
    return 0.5 * x**2 - x * y

# The total derivative of the objective w.r.t. y is 4y - 2
# The partial derivative of the objective w.r.t y is 2y


def objective(y, x):
    return (x - 1)**2 + y**2


def run():
    x = torch.tensor([0.], requires_grad=True, dtype=torch.float)
    y = torch.tensor([2.], requires_grad=True, dtype=torch.float)

    optimizer = optim.LBFGS([x], lr=1e-1, max_iter=20, history_size=100)
    max_epoch = 5
    for epoch in range(max_epoch):
        def closure():
            optimizer.zero_grad()
            loss = loss_function(y, x)
            loss.backward(create_graph=True, retain_graph=True)
            return loss

        optimizer.step(closure)
        print("loss is", loss_function(y, x))

    # Only get partial derivative, rather than total derivative
    J = torch.autograd.grad(objective(y, x), y)[0]
    J = J.detach().numpy()
    print("J is", J)
    print("optimized x is", x[0].data.numpy())


if __name__ == '__main__':
    run()
