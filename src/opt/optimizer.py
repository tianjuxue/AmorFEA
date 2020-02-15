import torch


class Optimizer(object):

    def __init__(self, args):
        self.args = args

    def _objective(self, x):
        x = torch.tensor(x, dtype=torch.float)
        L = self._obj(x)
        return L.data.numpy()

    def _derivative(self, x):
        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        L = self._obj(x)
        J = torch.autograd.grad(L, x, create_graph=True,
                                retain_graph=True)[0]
        J = J.detach().numpy().flatten()
        return J
