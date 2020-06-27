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
from ..ml.models import MLP_2
from ipopt import minimize_ipopt


class OptimizerDolfin(Optimizer):
    """
    Level 1
    """

    def __init__(self, args):
        super(OptimizerDolfin, self).__init__(args)
        self.poisson = PoissonDolfin(self.args)


class OptimizerDolfinReconstruction(OptimizerDolfin):
    """
    Level 2
    """

    def __init__(self, args):
        super(OptimizerDolfinReconstruction, self).__init__(args)
        self.target_dof, self.target_u, self.perfect_init_guess = target_solution_rc(
            self.args, self.poisson)
        self.args.input_size = self.poisson.num_dofs
        self.alpha = 1e-3

    def optimize(self):
        x_initial = 1e-0*np.random.randn(self.args.input_size)
        # x_initial = np.zeros(self.args.input_size)
        # x_initial = self.perfect_init_guess

        options = {'eps': 1e-15, 'maxiter': 1000,
                   'disp': True}  # CG > BFGS > Newton-CG
        start = time.time()
        res = opt.minimize(fun=self._objective,
                           x0=x_initial,
                           method='CG',
                           jac=self._derivative,
                           callback=None,
                           options=options)

        end = time.time()
        time_elapsed = end - start
        print("Time elasped {}".format(time_elapsed))
        return res.x, time_elapsed, res.nfev


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
            self.poisson.name + '/model_mlp_2'
        self.model = MLP_2(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))
        self.B_sp = self.trainer.B_sp.to_dense()
        self.A_sp = self.trainer.A_sp.to_dense()

        # self.model.adj = self.model.adj.to_dense()
        self.model.B_sp = self.trainer.B_sp.to_dense()


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
        L_diff = diff * tmp1
        # diff = (solution - self.target_dof)**2

        tmp2 = batch_mat_vec(self.B_sp, source)
        L_reg = source * tmp2

        L = 0.5 * (L_diff.sum() + self.alpha * L_reg.sum())
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
        L_tape = da.assemble((0.5 * fa.inner(u - self.target_u, u - self.target_u)
                              + 0.5 * self.alpha * fa.inner(f, f)) * fa.dx)
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
def target_solution_rc(args, pde):
    pde.source = da.Expression("k*100*exp( (-(x[0]-x0)*(x[0]-x0) -(x[1]-x1)*(x[1]-x1)) / (2*0.01*l) )",
                               k=1, l=1, x0=0.1, x1=0.1, degree=3)
    source_vec = da.interpolate(pde.source, pde.V)
    save_solution(args, source_vec, 'opt_fem_f')
    u = pde.solve_problem_variational_form()
    dof_data = torch.tensor(u.vector()[:], dtype=torch.float).unsqueeze(0)
    return dof_data, u, source_vec.vector()[:]


def produce_solution(pde, x):
    pde.set_control_variable(x)
    u = pde.solve_problem_variational_form()
    return u


def run_rec(args, seed):
    print("seed {}".format(seed))
    alpha_list = [1e-6, 1e-3, 1e-0] 
    optimizer_nn = OptimizerDolfinReconstructionSurrogate(args)
    optimizer_ad = OptimizerDolfinReconstructionAdjoint(args)
    np.random.seed(seed)
    objective_nn = []
    objective_ad = []
    time_nn = []
    time_ad = []
    for alpha in alpha_list:
        optimizer_nn.alpha = alpha
        x_nn, t_nn, _ = optimizer_nn.optimize()
        scalar_field_paraview(args, x_nn, optimizer_nn.poisson, "opt_nn_f")

        optimizer_ad.alpha = alpha
        x_ad, t_ad, _ = optimizer_ad.optimize()
        scalar_field_paraview(args, x_ad, optimizer_ad.poisson, "opt_ad_f")

        # To compute true objective, use optimizer_ad for both
        obj_nn = optimizer_ad._objective(x_nn)
        obj_ad = optimizer_ad._objective(x_ad)

        objective_nn.append(obj_nn)
        objective_ad.append(obj_ad)
        time_nn.append(t_nn)
        time_ad.append(t_ad) 


    objective_nn = np.asarray(objective_nn)
    objective_ad = np.asarray(objective_ad)
    time_nn = np.asarray(time_nn)
    time_ad = np.asarray(time_ad)

    print("true optimized objective nn", objective_nn)
    print("true optimized objective ad", objective_ad)
    print("time nn", time_nn)
    print("time ad", time_ad)

    np.save('data/numpy/dolfin/opt/seed{}/objective_nn.npy'.format(seed), objective_nn)
    np.save('data/numpy/dolfin/opt/seed{}/objective_ad.npy'.format(seed), objective_ad)
    np.save('data/numpy/dolfin/opt/seed{}/time_nn.npy'.format(seed), time_nn)
    np.save('data/numpy/dolfin/opt/seed{}/time_ad.npy'.format(seed), time_ad)

    return objective_nn, objective_ad, time_nn, time_ad
 

def run(args):
    objective_nn_collect = []
    objective_ad_collect = []
    time_nn_collect = []
    time_ad_collect = []
    for i, seed in enumerate(range(2, 7, 1)):
        print("\n\nround {}".format(i))
        # objective_nn, objective_ad, time_nn, time_ad = run_rec(args, seed)
        objective_nn = np.load('data/numpy/dolfin/opt/seed{}/objective_nn.npy'.format(seed))
        objective_ad = np.load('data/numpy/dolfin/opt/seed{}/objective_ad.npy'.format(seed))
        time_nn = np.load('data/numpy/dolfin/opt/seed{}/time_nn.npy'.format(seed))
        time_ad = np.load('data/numpy/dolfin/opt/seed{}/time_ad.npy'.format(seed))        
        objective_nn_collect.append(objective_nn)
        objective_ad_collect.append(objective_ad)
        time_nn_collect.append(time_nn)
        time_ad_collect.append(time_ad)
    objective_nn_collect = np.asarray(objective_nn_collect)
    objective_ad_collect = np.asarray(objective_ad_collect)
    time_nn_collect = 1000*np.asarray(time_nn_collect)
    time_ad_collect = 1000*np.asarray(time_ad_collect)

    for i in range(3):
        print("\n")
        print("mean of nn obj is {:.6f}, std is {:.6f}, for alpha {}".format(
                    objective_nn_collect[:, i].mean(),
                    objective_nn_collect[:, i].std(), i))
        print("mean of ad obj is {:.6f}, std is {:.6f}, for alpha {}".format(
                    objective_ad_collect[:, i].mean(),
                    objective_ad_collect[:, i].std(), i))
        print("mean of nn time is {:.1f}, std is {:.1f}, for alpha {}".format(
                    time_nn_collect[:, i].mean(),
                    time_nn_collect[:, i].std(), i))
        print("mean of ad time is {:.1f}, std is {:.1f}, for alpha {}".format(
                    time_ad_collect[:, i].mean(),
                    time_ad_collect[:, i].std(), i))


if __name__ == '__main__':
    args = arguments.args
    np.set_printoptions(precision=10)
    run(args)
    # run_rec(args, 2)

 
