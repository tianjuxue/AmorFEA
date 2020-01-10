import time
import torch
import numpy as np
import scipy.optimize as opt
import fenics as fa
import dolfin_adjoint as da
from .optimizer import Optimizer
from .. import arguments
from ..graph.visualization import scalar_field_paraview, save_solution
from ..pde.poisson_dolfin import PoissonDolfin
from ..ml.trainer import batch_mat_vec
from ..ml.trainer_dolfin import TrainerDolfin
from ..ml.models import MLP, MixedNetwork


class OptimizerDolfin(Optimizer):
    def __init__(self, args):
        super(OptimizerDolfin, self).__init__(args)

        self.trainer = TrainerDolfin(args)
        self.path = self.args.root_path + '/' + self.args.model_path + '/' + \
                    self.trainer.poisson.name + '/model_10_2'
        self.model = MixedNetwork(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))
        self.model.B_sp = self.trainer.B_sp.to_dense()
        self.model.adj = self.model.adj.to_dense()

        self.target = target_solution(self.args, self.trainer.poisson)

    def optimize(self):
        x_initial = np.zeros(self.args.input_size)
        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True} # CG > BFGS > Newton-CG
        res = opt.minimize(fun=self._objective,
                           x0=x_initial, 
                           method='CG', 
                           jac=self._derivative,
                           callback=None,
                           options=options)
        x_opt = res.x.reshape(-1, self.args.input_size)
        source = torch.tensor(x_opt, dtype=torch.float)
        solution = self.model(source)
        return res.x
 
    # def _obj(self, source):
    #     solution = self.model(source)
    #     diff = solution - self.target
    #     tmp = batch_mat_vec(self.model.B_sp, diff)
    #     L_diff = diff*tmp
    #     alpha = 0*1e-5
    #     tmp = batch_mat_vec(self.model.B_sp, source)
    #     L_ref = source*tmp   
    #     L = L_diff.sum() + alpha*L_ref.sum()  
    #     return L

    def _obj(self, source):
        solution = self.model(source)
        diff = (solution - self.target)**2
        L = diff.sum()       
        return L

    def _objective(self, x):
        source = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        L = self._obj(source)
        return L.data.numpy()

    def _derivative(self, x):
        source = torch.tensor(x, dtype=torch.float, requires_grad=True).unsqueeze(0)
        L = self._obj(source)

        start = time.time()
        J = torch.autograd.grad(L, source, create_graph=True,
                                   retain_graph=True)[0]
        end = time.time()
        print("time elapsed", end - start)
        exit()

        J = J.detach().numpy().flatten()
        return J


'''Helpers'''
def target_solution(args, pde):
    pde.source = da.Expression(("100*sin(2*pi*x[0])"),  degree=3)
    u = pde.solve_problem_variational_form()
    save_solution(args, u, 'gt')
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)
    return dof_data
 
def produce_solution(pde, x):
    pde.set_control_variable(x)
    u = pde.solve_problem_variational_form()
    return u


if __name__ == '__main__':
    args = arguments.args
    optimizer_dolfin = OptimizerDolfin(args)
    # start = time.time()
    x = optimizer_dolfin.optimize()
    # end = time.time()
    # print("time elapsed", end - start)
    u = produce_solution(optimizer_dolfin.trainer.poisson, x)
    save_solution(args, u, 'opt')