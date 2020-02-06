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
        self.target_point = np.array([0, -2])
        self.para_data = None

    def _opt(self, alpha=1e-2, x_initial=None, maxiter=200, log_interval=20):
        if x_initial is None:
            x_initial = np.zeros(self.args.input_size)
 
        x = x_initial
        start = time.time()
        wall_time = [0]
        objective = []
        source = [x]
        for i in range(maxiter):
            obj = self._objective(x)
            der = self._derivative(x)
            x = x - alpha*der
            if i % log_interval == 0:
                print("loop {} obj {}".format(i, obj))
                wall_time.append(time.time()-start)
                objective.append(obj)
                source.append(x)            
        x_opt = x
        objective.append(self._objective(x))
        return x_opt, np.asarray(wall_time), np.asarray(objective), np.asarray(source)

    def L_dist(self, solution):
        L = (solution[0][self.tip_x1_index] - self.target_point[0])**2 \
           +(solution[0][self.tip_x2_index] - self.target_point[1])**2    
        return L     

    def evaluate(self, source):
        solution, _ = self.trainer.forward_prediction(source, model=self.model)
        L = self.L_dist(np.expand_dims(solution, axis=0)) 
        return L, solution

    def batch_evaluate(self, source):
        Ls = []
        sols = []
        for s in source:
            L, sol = self.evaluate(s)
            Ls.append(L)
            sols.append(sol)
            print("Evaluated L", L)
        return np.asarray(Ls), np.asarray(sols)


class OptimizerRobotPointFree(OptimizerRobotPoint):
    def __init__(self, args):
        super(OptimizerRobotPointFree, self).__init__(args)

    def optimize(self, x_initial=None):
        if x_initial is None:
            x_initial = 0.1*np.ones(self.args.input_size)
 
        x = x_initial
        self._obj(x)
        options = {'maxiter': 100, 'disp': True, 'adaptive': True} # CG > BFGS > Newton-CG
        res = opt.minimize(fun=self._obj,
                           x0=x_initial, 
                           method='Nelder-Mead', 
                           options=options)
        x_opt = x
        return x_opt

    def _obj(self, source):
        solution, _ = self.trainer.forward_prediction(source, model=None, para_data=self.para_data)         
        L = self.L_dist(torch.tensor(solution, dtype=torch.float).unsqueeze(0))
        print(L)
        return L.item()
 

class OptimizerRobotPointSurrogate(OptimizerRobotPoint):
    def __init__(self, args):
        super(OptimizerRobotPointSurrogate, self).__init__(args)

    def optimize(self, alpha=1e-2, x_initial=None, maxiter=100, log_interval=100):
        return self._opt(alpha=alpha, x_initial=x_initial, maxiter=maxiter, log_interval=log_interval)

    def _obj(self, source):
        source = source.unsqueeze(0)
        solution = self.model(source)
        L = self.L_dist(solution) 
        return L


class OptimizerRobotPointAdjoint(OptimizerRobotPoint):
    def __init__(self, args):
        super(OptimizerRobotPointAdjoint, self).__init__(args)

    def optimize(self, alpha=2*1e-2, x_initial=None, maxiter=20, log_interval=1):
        return self._opt(alpha=alpha, x_initial=x_initial, maxiter=maxiter, log_interval=log_interval)

    def _objective(self, source):
        _, self.para_data = self.trainer.forward_prediction(source, model=None, para_data=self.para_data)
        _, _, L = self._objective_partials(source, self.para_data) 
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


'''Helpers'''
def heart_shape():
    def x_para(t):
        return 16*np.sin(t)**3
    def y_para(t):
        return 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t) - 5
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


def circle_shape():
    t = np.linspace(0, np.pi, 4)
    x = 2*np.cos(t - np.pi/2.)
    y = 2*np.sin(t - np.pi/2.)
    return np.asarray([x, y])


def run_mixed_opt(alpha_nn,
                  alpha_ad,
                  maxiter_nn,
                  maxiter_ad,
                  log_interval_nn,
                  log_interval_ad,
                  optimizer_nn,
                  optimizer_ad
                  ):
    x_opt, wall_time_nn, objective_nn, source_nn = optimizer_nn.optimize(alpha=alpha_nn, 
                                                                         x_initial=None, 
                                                                         maxiter=maxiter_nn, 
                                                                         log_interval=log_interval_nn)
    solver = RobotSolver(optimizer_nn.args, optimizer_nn.trainer.graph_info)
    solver.reset_parameters_network(torch.tensor(x_opt, dtype=torch.float).unsqueeze(0), optimizer_nn.model)
    para_data = solver.para.data
    optimizer_ad.para_data = para_data 
    x_opt, wall_time_ad, objective_ad, source_ad = optimizer_ad.optimize(alpha=alpha_ad, 
                                                                         x_initial=x_opt, 
                                                                         maxiter=maxiter_ad, 
                                                                         log_interval=log_interval_ad)

    wall_time_mix = np.concatenate((wall_time_nn, wall_time_ad[1:] + wall_time_nn[-1]))
    objective_mix = np.concatenate((objective_nn, objective_ad[1:]))
    source_mix = np.concatenate((source_nn, source_ad[1:]))
    return x_opt, wall_time_mix, objective_mix, source_mix


def run_single_opt(alpha,
                   maxiter,
                   log_interval,
                   optimizer
               ):
    x_opt, wall_time, objective, source = optimizer.optimize(alpha=alpha, 
                                                             x_initial=None, 
                                                             maxiter=maxiter, 
                                                             log_interval=log_interval)

    return x_opt, wall_time, objective, source


def run_one_case(args,
                 alpha_nn,
                 alpha_ad1,
                 alpha_ad2,
                 maxiter_nn,
                 maxiter_ad1,
                 maxiter_ad2,
                 log_interval_nn,
                 log_interval_ad1,
                 log_interval_ad2,
                 target_point,
                 case_number
                 ):

    print("\ncase number {}".format(case_number))

    optimizer_nn = OptimizerRobotPointSurrogate(args)
    optimizer_nn.target_point = target_point  
    optimizer_ad = OptimizerRobotPointAdjoint(args)
    optimizer_ad.target_point = target_point

    # x_nn, wall_time_nn, objective_nn, source_nn = run_single_opt(alpha_nn, maxiter_nn, log_interval_nn, optimizer_nn)
    _, wall_time_ad, objective_ad, source_ad = run_single_opt(alpha_ad1, maxiter_ad1, log_interval_ad1, optimizer_ad)
    print("\n")
    _, wall_time_mix, objective_mix, source_mix = run_mixed_opt(alpha_nn,
                                                                alpha_ad2,
                                                                maxiter_nn,
                                                                maxiter_ad2,
                                                                log_interval_nn,
                                                                log_interval_ad2,
                                                                optimizer_nn,
                                                                optimizer_ad)


    nn_number = maxiter_nn//log_interval_nn
    objective_mix[:nn_number + 1], _ = optimizer_nn.batch_evaluate(source_mix[:nn_number + 1])
    _, optimal_solution = optimizer_nn.evaluate(source_mix[-1])
    
    # print("true error nn", objective_nn[-1])
    print("true error ad", objective_ad[-1])
    print("true error mix", objective_mix[-1])

    np.savez(args.root_path + '/' + args.numpy_path 
            + '/robot/deploy/case' + str(case_number) + '.npz',  
            wall_time_ad=wall_time_ad,
            objective_ad=objective_ad,
            nn_number=nn_number,
            wall_time_mix=wall_time_mix,
            objective_mix=objective_mix,
            target_point=target_point
            )
    scalar_field_paraview(args, optimal_solution, optimizer_nn.trainer.poisson, "/robot/deploy/u" + str(case_number))


def run_walltime(args):
    # Manully tuned best parameter
    # target_coos = heart_shape()
    # target_coos = target_coos[:, [3,7,11,15]]

    target_coos = circle_shape()

    alpha_ad1_list = [1e-2, 1e-2, 2*1e-3, 2*1e-3]
    alpha_nn_list = [1e-2, 1e-2, 2*1e-3, 2*1e-3]
    alpha_ad2_list = [1e-2, 1e-2, 2*1e-3, 2*1e-3]

    maxiter_ad1_list = [20, 20, 20, 20]
    maxiter_nn_list = [400, 400, 4000, 6000]
    maxiter_ad2_list = [20, 20, 20, 20]
    
    log_interval_ad1_list = [1, 1, 1, 1]  
    log_interval_nn_list = [40, 40, 400, 600]
    log_interval_ad2_list = [1, 1, 1, 1]

    for i in range(3, 4):
        run_one_case(args,
                     alpha_nn_list[i],
                     alpha_ad1_list[i],
                     alpha_ad2_list[i],
                     maxiter_nn_list[i],
                     maxiter_ad1_list[i],
                     maxiter_ad2_list[i] ,
                     log_interval_nn_list[i],
                     log_interval_ad1_list[i],
                     log_interval_ad2_list[i],
                     target_coos[:, i],
                     i)


def run_step(args):
    target_coos = circle_shape()
    alpha_list = [1e-2, 1e-2, 2*1e-3, 2*1e-3]
    for case_number in range(2, 4):
        optimizer_nn = OptimizerRobotPointSurrogate(args)
        optimizer_ad = OptimizerRobotPointAdjoint(args)
        print("case_number", case_number)
        target_point = target_coos[:, case_number]
        optimizer_nn.target_point = target_point
        optimizer_ad.target_point = target_point     

        _, wall_time_ad, objective_ad, source_ad = run_single_opt(alpha_list[case_number], 100, 1, optimizer_ad)
        _, wall_time_nn, objective_nn, source_nn = run_single_opt(alpha_list[case_number], 100, 1, optimizer_nn)
        objective_nn, _ = optimizer_nn.batch_evaluate(source_nn)
        np.savez(args.root_path + '/' + args.numpy_path 
                 + '/robot/deploy/case_step' + str(case_number) + '.npz',  
                 objective_ad=objective_ad,
                 objective_nn=objective_nn,
                 target_point=target_point,
                 wall_time_ad=wall_time_ad,
                 wall_time_nn=wall_time_nn
                )

def run_gradient_free(args):
    target_coos = circle_shape()
    target_point = target_coos[:, 1]
    optimizer_fr = OptimizerRobotPointFree(args)
    optimizer_fr.target_point = target_point 
    optimizer_fr.optimize()


if __name__ == '__main__':
    args = arguments.args
    run_step(args)