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
from ..ml.models import MLP, MixedNetwork, LinearRegressor


class OptimizerDolfin(Optimizer):
    def __init__(self, args):
        super(OptimizerDolfin, self).__init__(args)

        self.trainer = TrainerDolfin(args)
        self.path = self.args.root_path + '/' + self.args.model_path + '/' + \
                    self.trainer.poisson.name + '/model_0'
        self.model = MLP(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))
        self.B_sp = self.trainer.B_sp.to_dense()
        self.A_sp = self.trainer.A_sp.to_dense()

        # self.model.adj = self.model.adj.to_dense()
        self.model.B_sp = self.trainer.B_sp.to_dense()
        self.target_dof, self.target_u = target_solution(self.args, self.trainer.poisson)

    def optimize(self):
        x_initial = np.array([0.5, 0.5])
        # x_initial = np.random.randn(self.args.input_size)
        # x_initial = np.ones(self.args.input_size)
        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True} # CG > BFGS > Newton-CG

        start = time.time()
        res = opt.minimize(fun=self._objective,
                           x0=x_initial, 
                           method='CG', 
                           jac=self._derivative,
                           callback=None,
                           options=options)
        end = time.time()
        print("Time elasped {}".format(end - start))
        # x_opt = res.x.reshape(-1, self.args.input_size)
        # source = torch.tensor(x_opt, dtype=torch.float)
        # solution = self.model(source)
        return res.x
 
    def _objective(self, x):
        L, _, _ = self.trainer.poisson.adjoint_obj(x, self.target_u)
        return L

    def _derivative(self, x):
        J = self.trainer.poisson.adjoint_der(x, self.target_u)
        return J

    def _obj(self, p):
        k=100
        l=0.01
        coo = torch.tensor(self.trainer.poisson.coo_dof, dtype=torch.float)
        source = k*torch.exp( (-(coo[:, 0] - p[0])**2 -(coo[:, 1]- p[1])**2) / (2*l) )
        solution = self.model(source.unsqueeze(0))
        diff = solution - self.target_dof

        tmp = batch_mat_vec(self.B_sp, diff)
        L_diff = diff*tmp
        # diff = (solution - self.target_dof)**2

        L = L_diff.sum()   
        return L

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

    def debug(self):
        truth_map = []
        coo = torch.tensor(self.trainer.poisson.coo_dof, dtype=torch.float)
        for i in range(self.args.input_size):
            L = self._obj(coo[i])
            truth_map.append(L.data.numpy())
        truth_map = np.array(truth_map)
        scalar_field_paraview(self.args, truth_map, self.trainer.poisson, "opt_tm")

 

'''Helpers'''
def target_solution(args, pde):
    pde.source = da.Expression("k*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*l) )", 
                               k=100, l=0.01, x0=0.9, x1=0.1, degree=3)
    # pde.source = da.interpolate(pde.source, pde.V)
    # save_solution(args, pde.source, 'opt_fem_f')

    u = pde.solve_problem_variational_form()
    save_solution(args, u, 'opt_fem_u')
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)

    return dof_data, u

def produce_solution(pde, x):
    pde.set_control_variable(x)
    u = pde.solve_problem_variational_form()
    return u





if __name__ == '__main__':
    args = arguments.args
    optimizer_dolfin = OptimizerDolfin(args)
    # start = time.time()
 
    x = optimizer_dolfin.optimize()
    print(x)
 
    # u = produce_solution(optimizer_dolfin.trainer.poisson, x)
    # save_solution(args, u, 'opt_nn_u')
    # scalar_field_paraview(args, x, optimizer_dolfin.trainer.poisson, "opt_nn_f")