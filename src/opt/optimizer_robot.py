import numpy as np
import torch
import scipy.optimize as opt
import time
import matplotlib.pyplot as plt
from .optimizer import Optimizer
from ..ml.trainer_robot import TrainerRobot
from ..ml.models import RobotNetwork, RobotSolver
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class OptimizerRobot(Optimizer):
    def __init__(self, args):
        super(OptimizerRobot, self).__init__(args)
        self.tip_x1_index = 6
        self.tip_x2_index = 7
        self.trainer = TrainerRobot(args, opt=True)
        self.path = self.args.root_path + '/' + self.args.model_path + '/' + \
                    self.trainer.poisson.name + '/model_sss'
        self.model = RobotNetwork(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))


class OptimizerRobotTrajectory(OptimizerRobot):
    def __init__(self, args):
        super(OptimizerRobotTrajectory, self).__init__(args)
        self.target_coos = heart_shape()
        self.n_pts = self.target_coos.shape[1]


    def optimize(self):
        x_initial = np.zeros(self.args.input_size * self.n_pts)
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
        print("NN surrogate, loss is", self.trainer.loss_function(source, solution).data.numpy())
        for i in range(31):
            scalar_field_paraview(self.args, solution.data.numpy()[i], self.trainer.poisson, "/robot/time_series_nn/u" + str(i))

        for i in range(31):
            gt_sol = self.trainer.forward_prediction(x_opt[i], self.model)
            scalar_field_paraview(self.args, gt_sol, self.trainer.poisson, "/robot/time_series_gt/u" + str(i))

        return res.x
    
    def _obj(self, source):
        source = source.reshape(-1, self.args.input_size)
        solution = self.model(source)
        sol_tip = solution[:, [self.tip_x1_index, self.tip_x2_index]]
        tar_tip = torch.tensor(self.target_coos.transpose(), dtype=torch.float)
        L_dist = ((sol_tip - tar_tip)**2).sum()
        L_reg = ((source[1:,:] - source[:-1, :])**2).sum()
        alpha = 0*1e-3
        L = L_dist + alpha*L_reg
        return L


class OptimizerRobotPoint(OptimizerRobot):
    def __init__(self, args):
        super(OptimizerRobotPoint, self).__init__(args)
        # self.target_x1 = -0
        # self.target_x2 = -2
        self.target_x1 = 1
        self.target_x2 = 0
        self.para_data = None

    def _opt(self, alpha=1e-2, x_initial=None, max_iter=1000, log_interval=100):
        if x_initial is None:
            x_initial = np.zeros(self.args.input_size)
 
        x = x_initial
        start = time.time()
        wall_time = []
        objective = []
        source = []
        for i in range(max_iter):
            obj = self._objective(x)
            der = self._derivative(x)
            x = x - alpha*der
            if i % log_interval == 0:
                print("loop {} obj {}\n".format(i, obj))
                wall_time.append(time.time()-start)
                objective.append(obj)
                source.append(x)
        x_opt = x
        # x_opt = x_opt.reshape(-1, self.args.input_size)
        # source = torch.tensor(x_opt, dtype=torch.float)
        # solution = self.model(source)
        return x_opt, np.asarray(wall_time), np.asarray(objective), np.asarray(source)

    def L_dist(self, solution):
        L = (solution[0][self.tip_x1_index] - self.target_x1)**2 \
           +(solution[0][self.tip_x2_index] - self.target_x2)**2    
        return L     

    def evaluate(self, source):
        solution, _ = self.trainer.forward_prediction(source, model=self.model)
        L = self.L_dist(np.expand_dims(solution, axis=0)) 
        return L

    def batch_evaluate(self, source):
        L = [self.evaluate(s) for s in source]
        return np.asarray(L)

class OptimizerRobotPointSurrogate(OptimizerRobotPoint):
    def __init__(self, args):
        super(OptimizerRobotPointSurrogate, self).__init__(args)

    def optimize(self):
        return self._opt(alpha=1e-2, x_initial=None, max_iter=1000, log_interval=100)

    def _obj(self, source):
        source = source.unsqueeze(0)
        solution = self.model(source)
        L = self.L_dist(solution) 
        return L


class OptimizerRobotPointAdjoint(OptimizerRobotPoint):
    def __init__(self, args):
        super(OptimizerRobotPointAdjoint, self).__init__(args)

    def optimize(self, alpha=2*1e-2, x_initial=None):
        return self._opt(alpha=alpha, x_initial=x_initial, max_iter=20, log_interval=1)

    def _objective(self, source):
        _, self.para_data = self.trainer.forward_prediction(source, model=None, para_data=self.para_data)
        _, _, L = self._objective_partials(source, self.para_data) 
        # print("distance loss is", L)
        return L

    def _derivative(self, source):
        dcdx, dcdy = self._constraint_partials(source, self.para_data)
        dLdx, dLdy, _ = self._objective_partials(source, self.para_data) 
        J = self._adjoint_derivative(dcdx, dcdy, dLdx, dLdy)
        return J

    def _adjoint_derivative(self, dcdx, dcdy, dLdx, dLdy):
        dcdx_T = dcdx.transpose()
        adjoint_sol = np.linalg.solve(dcdx_T, dLdx) 
        total_derivative = -np.matmul(adjoint_sol, dcdy) + dLdy
        return total_derivative

    def _objective_partials(self, source, para_data):  
        solver = RobotSolver(self.args, self.trainer.graph_info)
        solver.reset_parameters_data(para_data)
        
        source = torch.tensor(source, requires_grad=True, dtype=torch.float)
        source_input = source.unsqueeze(0)

        solution = solver(source_input)
        L = self.L_dist(solution) 

        dLdx = torch.autograd.grad(L, solver.para, create_graph=True, retain_graph=True)[0]
        dLdy = torch.autograd.grad(L, source, create_graph=True, retain_graph=True)[0]

        return dLdx.data.numpy(), dLdy.data.numpy(), L.data.numpy()

    def _constraint_partials(self, source, para_data):
        solver = RobotSolver(self.args, self.trainer.graph_info)
        solver.reset_parameters_data(para_data)

        source = torch.tensor(source, requires_grad=True, dtype=torch.float)
        source_input = source.unsqueeze(0)

        solution = solver(source_input)
        L = self.trainer.loss_function(source_input, solution)
        c = torch.autograd.grad(L, solver.para, create_graph=True, retain_graph=True)[0]

        dcdx = torch.stack([torch.autograd.grad(c[i], solver.para, create_graph=True, retain_graph=True)[0] for i in range(len(c))])
        dcdy = torch.stack([torch.autograd.grad(c[i], source, create_graph=True, retain_graph=True)[0] for i in range(len(c))])

        return dcdx.data.numpy(), dcdy.data.numpy()        


class OptimizerRobotPointMixed(OptimizerRobotPoint):

    def optimize(self):
        # self.optimize(method_flag=1)
          
        optimizer_nn = OptimizerRobotPointSurrogate(self.args)
        x_opt, wall_time_nn, objective_nn, source_nn = optimizer_nn.optimize()
        solver = RobotSolver(self.args, self.trainer.graph_info)
        solver.reset_parameters_network(torch.tensor(x_opt, dtype=torch.float).unsqueeze(0), optimizer_nn.model)
        para_data = solver.para.data

        optimizer_ad = OptimizerRobotPointAdjoint(self.args)
        optimizer_ad.para_data = para_data 
        x_opt, wall_time_ad, objective_ad, source_ad = optimizer_ad.optimize(alpha=1e-2, x_initial=x_opt)

        wall_time_mix = np.concatenate((wall_time_nn, wall_time_ad + wall_time_nn[-1]))
        objective_mix = np.concatenate((objective_nn, objective_ad))
        source_mix = np.concatenate((source_nn, source_ad))

        return x_opt, wall_time_mix, objective_mix, source_mix

'''Helpers'''
def x_para(t):
    return 16*np.sin(t)**3

def y_para(t):
    return 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t) - 5

def heart_shape():
    vertical_dist = 2
    norm_factor = vertical_dist / (y_para(0) - y_para(np.pi))
    t = np.linspace(0, 2*np.pi, 31)
    x = norm_factor*x_para(t)
    y = norm_factor*y_para(t)
    return np.asarray([x, y])

def plot_hs():
    x, y = heart_shape()
    fig = plt.figure(0)
    plt.tick_params(labelsize=14)
    # plt.xlabel('xlabel')
    # plt.ylabel('ylabel')
    # plt.legend(loc='upper left')
    plt.plot(x, y)


def run():
    optimizer_nn = OptimizerRobotPointSurrogate(args)
    optimizer_ad = OptimizerRobotPointAdjoint(args)
    optimizer_mix = OptimizerRobotPointMixed(args)

    # x_nn, wall_time_nn, objective_nn = optimizer_nn.optimize()
    x_ad, wall_time_ad, objective_ad, source_ad = optimizer_ad.optimize(alpha=1e-2)
    x_mix, wall_time_mix, objective_mix, source_mix = optimizer_mix.optimize()

    # print("true error nn", optimizer_nn.evaluate(x_nn))
    print("true error ad", optimizer_ad.evaluate(x_ad))
    print("true error mix", optimizer_mix.evaluate(x_mix))

    nn_number = 10
    objective_mix[:nn_number] = optimizer_mix.batch_evaluate(source_mix[:nn_number])

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(wall_time_ad, objective_ad, linestyle='--', marker='o', color='blue')
    ax.plot(wall_time_mix[:nn_number], objective_mix[:nn_number], linestyle='--', marker='o', color='red')
    ax.plot(wall_time_mix[nn_number:], objective_mix[nn_number:], linestyle='--', marker='o', color='blue')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()


if __name__ == '__main__':
    args = arguments.args
    # plot_hs()
    run()