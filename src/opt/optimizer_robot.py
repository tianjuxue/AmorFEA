import numpy as np
import torch
import scipy.optimize as opt
from ..ml.trainer_robot import TrainerRobot
from ..ml.models import RobotNetwork
from .. import arguments
from ..graph.visualization import scalar_field_paraview
from .trajectories import heart_shape


class Optimizer(object):
    def __init__(self, args):
        self.args = args


class RobotOptimizer(Optimizer):
    def __init__(self, args):
        super(RobotOptimizer, self).__init__(args)
        self.target_x1 = -0
        self.target_x2 = -2
        self.tip_x1_index = 6
        self.tip_x2_index = 7
        self.target_coos = heart_shape()
        self.n_pts = self.target_coos.shape[1]
        self.trainer_robot = TrainerRobot(args, opt=True)
        self.path = self.args.root_path + '/' + self.args.model_path + '/robot/model_sss'
        self.model = RobotNetwork(self.args, self.trainer_robot.graph_info)
        self.model.load_state_dict(torch.load(self.path))


class RobotOptimizerTrajectory(RobotOptimizer):
    def __init__(self, args):
        super(RobotOptimizerTrajectory, self).__init__(args)

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
        print("NN surrogate, loss is", self.trainer_robot.loss_function(source, solution).data.numpy())
        for i in range(31):
            scalar_field_paraview(self.args, solution.data.numpy()[i], self.trainer_robot.poisson, "time_series_nn/u" + str(i))

        for i in range(31):
            gt_sol = self.trainer_robot.forward_prediction(x_opt[i], self.model)
            scalar_field_paraview(self.args, gt_sol, self.trainer_robot.poisson, "time_series_gt/u" + str(i))

        return res.x
    
    def _obj(self, source):
        solution = self.model(source)
        sol_tip = solution[:, [self.tip_x1_index, self.tip_x2_index]]
        tar_tip = torch.tensor(self.target_coos.transpose(), dtype=torch.float)
        L_dist = ((sol_tip - tar_tip)**2).sum()
        L_reg = ((source[1:,:] - source[:-1, :])**2).sum()
        alpha = 1e-5
        L = L_dist + alpha*L_reg
        return L

    def _objective(self, x):
        x = x.reshape(-1, self.args.input_size)
        source = torch.tensor(x, dtype=torch.float)
        L = self._obj(source)
        return L.data.numpy()

    def _derivative(self, x):
        x = x.reshape(-1, self.args.input_size)
        source = torch.tensor(x, dtype=torch.float,  requires_grad=True)
        L = self._obj(source)
        J = torch.autograd.grad(L, source, create_graph=True,
                                   retain_graph=True)[0]
        J = J.detach().numpy().flatten()
        return J


class RobotOptimizerPoint(RobotOptimizer):
    def __init__(self, args):
        super(RobotOptimizerPoint, self).__init__(args)

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
        solution = self.model(source)
        L = (solution[0][self.tip_x1_index] - self.target_x1)**2 \
           +(solution[0][self.tip_x2_index] - self.target_x2)**2 
        return L

    def _objective(self, x):
        source = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        L = self._obj(source)
        return L.data.numpy()

    def _derivative(self, x):
        source = torch.tensor(x, dtype=torch.float, requires_grad=True).unsqueeze(0)
        L = self._obj(source)
        J = torch.autograd.grad(L, source, create_graph=True,
                                   retain_graph=True)[0]
        J = J.detach().numpy().flatten()
        return J


if __name__ == '__main__':
    args = arguments.args
    robot_optimizer = RobotOptimizerTrajectory(args)
    robot_optimizer.optimize()