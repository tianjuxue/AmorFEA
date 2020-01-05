import numpy as np
import torch
import scipy.optimize as opt
from ..ml.trainer_robot import TrainerRobot
from ..ml.models import RobotNetwork
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class Optimizer(object):
    def __init__(self, args):
        self.args = args


class RobotOptimizer(Optimizer):
    def __init__(self, args):
        super(RobotOptimizer, self).__init__(args)
        self.target_x1 = -0
        self.target_x2 = -2

    def _objective(self, x):

        source = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        solution = model(source)
        solution = solution.data.numpy().flatten()
        square_dist = (solution[0] - self.target_x1)**2 + (solution[1] - self.target_x2)**2 
        return square_dist

    def _objective_der(self, x):
        source = torch.tensor(x, dtype=torch.float, requires_grad=True).unsqueeze(0)
        solution = model(source)
        square_dist = (solution[0][0] - self.target_x1)**2 + (solution[0][1] - self.target_x2)**2 
        J = torch.autograd.grad(square_dist, source, create_graph=True,
                                   retain_graph=True)[0]
        J = J.detach().numpy().flatten()
        return J

    def _call_back(self, x):
        pass

if __name__ == '__main__':
    args = arguments.args

    trainer_robot = TrainerRobot(args, opt=True)
    path = args.root_path + '/' + args.model_path + '/robot/model_s'
    model = RobotNetwork(args, trainer_robot.graph_info)
    model.load_state_dict(torch.load(path))

    robot_optimizer = RobotOptimizer(args)

    x_initial = np.zeros(args.input_size)

    options={'eps': 1e-15, 'maxiter': 1000, 'disp': True}
    res = opt.minimize(robot_optimizer._objective,
                       x_initial, 
                       method='CG', 
                       jac=robot_optimizer._objective_der,
                       callback=robot_optimizer._call_back,
                       options=options)

    source = torch.tensor(res.x, dtype=torch.float).unsqueeze(0)
    solution = model(source)
    scalar_field_paraview(args, solution.data.numpy().flatten(), trainer_robot.poisson, "this")

    np.save('tmp.npy', res.x)
    print(res.x)

    # CG > BFGS > Newton-CG