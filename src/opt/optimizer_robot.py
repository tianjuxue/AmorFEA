import numpy as np
import torch
import scipy.optimize as opt
from .optimizer import Optimizer
from ..ml.trainer_robot import TrainerRobot
from ..ml.models import RobotNetwork
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class OptimizerRobot(Optimizer):
    def __init__(self, args):
        super(OptimizerRobot, self).__init__(args)
        self.target_x1 = -0
        self.target_x2 = -2
        self.tip_x1_index = 6
        self.tip_x2_index = 7
        self.target_coos = heart_shape()
        self.n_pts = self.target_coos.shape[1]
        self.trainer = TrainerRobot(args, opt=True)
        self.path = self.args.root_path + '/' + self.args.model_path + '/' + \
                    self.trainer.poisson.name + '/model_sss'
        self.model = RobotNetwork(self.args, self.trainer.graph_info)
        self.model.load_state_dict(torch.load(self.path))


class OptimizerRobotTrajectory(OptimizerRobot):
    def __init__(self, args):
        super(OptimizerRobotTrajectory, self).__init__(args)

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

    def _obj(self, source):
        source = source.reshape(-1, self.args.input_size)
        solution = self.model(source)
        L = (solution[0][self.tip_x1_index] - self.target_x1)**2 \
           +(solution[0][self.tip_x2_index] - self.target_x2)**2 
        return L
        

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


if __name__ == '__main__':
    args = arguments.args
    # plot_hs()
    optimizer_robot = OptimizerRobotTrajectory(args)
    optimizer_robot.optimize()