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
from ipopt import minimize_ipopt


class OptimizerDolfin(Optimizer):
    """
    Level 1
    """
    def __init__(self, args):
        super(OptimizerDolfin, self).__init__(args)
        self.poisson = PoissonDolfin(self.args)


class OptimizerDolfinIdentification(OptimizerDolfin):
    """
    Level 2
    """
    def __init__(self, args):
        super(OptimizerDolfin, self).__init__(args)
        self.target_dof, self.target_u = target_solution_id(self.args, self.poisson)

    def optimize(self):
        x_initial = np.array([0.5, 0.5, 0.5, 0.5])
        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True} # CG > BFGS > Newton-CG
        start = time.time()
        res = opt.minimize(fun=self._objective,
                           x0=x_initial, 
                           method='BFGS', 
                           jac=self._derivative,
                           callback=None,
                           options=options)
        end = time.time()
        print("Time elasped {}".format(end - start))
        return res.x
 
    def debug(self):
        truth_map = []
        coo = torch.tensor(self.poisson.coo_dof, dtype=torch.float)
        for i in range(self.args.input_size):
            L = self._obj(coo[i])
            truth_map.append(L.data.numpy())
        truth_map = np.array(truth_map)
        scalar_field_paraview(self.args, truth_map, self.poisson, "opt_tm")


class OptimizerDolfinReconstruction(OptimizerDolfin):
    """
    Level 2
    """
    def __init__(self, args):
        super(OptimizerDolfinReconstruction, self).__init__(args)
        self.target_dof, self.target_u, self.perfect_init_guess = target_solution_rc(self.args, self.poisson)
        self.args.input_size = self.poisson.num_dofs
        self.alpha = 0*1e-6

    def optimize(self):
        x_initial = np.random.randn(self.args.input_size)
        # x_initial = self.perfect_init_guess
        x_initial = np.ones(self.args.input_size)
        options = {'eps': 1e-15, 'maxiter': 1000, 'disp': True} # CG > BFGS > Newton-CG
        start = time.time()
        # res = opt.minimize(fun=self._objective,
        #                    x0=x_initial, 
        #                    method='CG', 
        #                    jac=self._derivative,
        #                    callback=None,
        #                    options=options)
        res = minimize_ipopt(self._objective, x_initial, jac=self._derivative)
        end = time.time()
        print("Time elasped {}".format(end - start))
        # x_opt = res.x.reshape(-1, self.args.input_size)
        # source = torch.tensor(x_opt, dtype=torch.float)
        # solution = self.model(source)
        return res.x


class OptimizerDolfinAdjoint(OptimizerDolfin):
    """
    Level 2
    """
    def __init__(self, args):
        super(OptimizerDolfinAdjoint, self).__init__(args)


class OptimizerDolfinSurrogate(OptimizerDolfin):
    """
    Level 2
    """
    def __init__(self, args):
        super(OptimizerDolfinSurrogate, self).__init__(args)
        self.trainer = TrainerDolfin(args)
        self.path = self.args.root_path + '/' + self.args.model_path + '/' + \
                    self.poisson.name + '/model_0'
        self.model = MLP(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))
        self.B_sp = self.trainer.B_sp.to_dense()
        self.A_sp = self.trainer.A_sp.to_dense()

        # self.model.adj = self.model.adj.to_dense()
        self.model.B_sp = self.trainer.B_sp.to_dense()


class OptimizerDolfinIdentificationSurrogate(OptimizerDolfinIdentification, OptimizerDolfinSurrogate):
    """
    Level 3
    """
    def __init__(self, args):
        super(OptimizerDolfinIdentificationSurrogate, self).__init__(args)


    def _obj(self, p):
        coo = torch.tensor(self.poisson.coo_dof, dtype=torch.float)
        source = 100*p[2]*torch.exp( (-(coo[:, 0] - p[0])**2 -(coo[:, 1]- p[1])**2) / (2*0.01*p[3]) )
        solution = self.model(source.unsqueeze(0))
        diff = solution - self.target_dof

        tmp = batch_mat_vec(self.B_sp, diff)
        L_diff = diff*tmp
        # diff = (solution - self.target_dof)**2

        L = L_diff.sum()   
        return L


class OptimizerDolfinIdentificationAdjoint(OptimizerDolfinIdentification, OptimizerDolfinAdjoint):
    """
    Level 3
    """
    def __init__(self, args):
        super(OptimizerDolfinIdentificationAdjoint, self).__init__(args)

    def _obj(self, x):
        p = da.Constant(x)
        x =  fa.SpatialCoordinate(self.poisson.mesh)
        self.poisson.source = 100*p[2]*fa.exp( (-(x[0]-p[0])*(x[0]-p[0]) -(x[1]-p[1])*(x[1]-p[1])) / (2*0.01*p[3]) )
        u = self.poisson.solve_problem_variational_form()
        L_tape = da.assemble((0.5 * fa.inner(u - self.target_u, u - self.target_u)) * fa.dx)
        L = float(L_tape)
        return L, L_tape, p

    def _objective(self, x):
        L, _, _ = self._obj(x)
        return L

    def _derivative(self, x):
        _, L_tape, p = self._obj(x)
        control = da.Control(p)
        J_tape = da.compute_gradient(L_tape, control)
        J = J_tape.values()
        return J


class OptimizerDolfinReconstructionSurrogate(OptimizerDolfinReconstruction, OptimizerDolfinSurrogate):
    """
    Level 3
    """
    def __init__(self, args):
        super(OptimizerDolfinReconstructionSurrogate, self).__init__(args)


    def _obj(self, source):
        source = source.unsqueeze(0)
        solution = self.model(source)
        diff = solution - self.target_dof

        tmp1 = batch_mat_vec(self.B_sp, diff)
        L_diff = diff*tmp1
        # diff = (solution - self.target_dof)**2

        tmp2 = batch_mat_vec(self.B_sp, source)
        L_reg = source*tmp2

        L = L_diff.sum() + self.alpha*L_reg.sum()
        return L


class OptimizerDolfinReconstructionAdjoint(OptimizerDolfinReconstruction, OptimizerDolfinAdjoint):
    """
    Level 3
    """
    def __init__(self, args):
        super(OptimizerDolfinReconstructionAdjoint, self).__init__(args)

    def _obj(self, x):
        f = da.Function(self.poisson.V)
        f.vector()[:] = x
        self.poisson.source = f
        u = self.poisson.solve_problem_variational_form()
        L_tape = da.assemble( ( 0.5 * fa.inner(u - self.target_u, u - self.target_u)  
                                +self.alpha*fa.inner(u, u) ) * fa.dx )
        L = float(L_tape)
        return L, L_tape, f

    def _objective(self, x):
        L, _, _ = self._obj(x)
        return L

    def _derivative(self, x):
        _, L_tape, f = self._obj(x)
        control = da.Control(f)
        J_tape = da.compute_gradient(L_tape, control)
        J = np.array(J_tape.vector()[:])
        return J


'''Helpers'''
def target_solution_id(args, pde):
    pde.source = da.Expression("k*100*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*0.01*l) )", 
                               k=1, l=1, x0=0.9, x1=0.1, degree=3)
    # pde.source = da.interpolate(pde.source, pde.V)
    # save_solution(args, pde.source, 'opt_fem_f')

    u = pde.solve_problem_variational_form()
    save_solution(args, u, 'opt_fem_u')
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)
    return dof_data, u

def target_solution_rc(args, pde):
    # pde.source = da.Expression(("100*sin(2*pi*x[0])"),  degree=3)
    pde.source = da.Expression("k*100*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*0.01*l) )", 
                               k=1, l=1, x0=0.9, x1=0.1, degree=3)
    debug_var = da.interpolate(pde.source, pde.V)
    # save_solution(args, pde.source, 'opt_fem_f')

    u = pde.solve_problem_variational_form()
    save_solution(args, u, 'opt_fem_u')
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)
    return dof_data, u, debug_var.vector()[:]

def produce_solution(pde, x):
    pde.set_control_variable(x)
    u = pde.solve_problem_variational_form()
    return u


if __name__ == '__main__':
    args = arguments.args
    optimizer = OptimizerDolfinReconstructionSurrogate(args)
    x = optimizer.optimize()
    # print(x)
 
    # u = produce_solution(optimizer.poisson, x)
    # save_solution(args, u, 'opt_nn_u')
    scalar_field_paraview(args, x, optimizer.poisson, "opt_ad_f")