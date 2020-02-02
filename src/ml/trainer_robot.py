'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Nonlinear case - Soft Robot
'''

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import collections
from .trainer import Trainer, batch_mat_vec, normalize_adj, boundary_flag_matrix
from .models import LinearRegressor, MLP, MixedNetwork, RobotNetwork, RobotSolver
from ..pde.poisson_robot import PoissonRobot
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class TrainerRobot(Trainer):
    def __init__(self, args, opt=False):
        super(TrainerRobot, self).__init__(args)
        self.poisson = PoissonRobot(self.args)
        self.opt = opt
        self.initialization()


    def loss_function(self, x_control, x_state, y_state=None):
        # y_state is not useful
        young_mod = 100
        poisson_ratio = 0.3
        shear_mod = young_mod / (2 * (1 + poisson_ratio))
        bulk_mod = young_mod / (3 * (1 - 2*poisson_ratio))

        F00 = batch_mat_vec(self.F00, x_state) + 1
        F01 = batch_mat_vec(self.F01, x_state)
        F10 = batch_mat_vec(self.F10, x_state)
        F11 = batch_mat_vec(self.F11, x_state) + 1

        J = F00*F11 - F01*F10
        Jinv = J**(-2 / 3)
        I1 = F00*F00 + F01*F01 + F10*F10 + F11*F11

        energy_density = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) +
                          (bulk_mod / 2) * (J - 1)**2)

        loss = (energy_density * self.weight_area).sum()

        return loss


    def initialization(self):
        bc_btm, bc_lx, bc_ly, bc_rx, bc_ry = self.poisson.boundary_flags_list
        interior = np.ones(self.poisson.num_dofs)
        for bc in self.poisson.boundary_flags_list:
            interior -= bc

        bc_btm_mat = boundary_flag_matrix(bc_btm)
        bc_lx_mat = boundary_flag_matrix(bc_lx)
        bc_ly_mat = boundary_flag_matrix(bc_ly)
        bc_rx_mat = boundary_flag_matrix(bc_rx)
        bc_ry_mat = boundary_flag_matrix(bc_ry)
        int_mat = boundary_flag_matrix(interior)

        self.bc_l_mat = bc_lx_mat
        self.bc_r_mat = bc_rx_mat
        self.args.input_size = self.bc_l_mat.shape[0] + self.bc_r_mat.shape[0]

        interior_size = int_mat.sum()
        control_bc_size = bc_lx_mat.sum() + bc_ly_mat.sum() + bc_rx_mat.sum() + bc_ry_mat.sum()
        shapes = [int(control_bc_size), int(interior_size)]

        lx_0, ly_0, rx_0, ry_0 = 0, 0, self.poisson.width, 0
        lx = np.matmul(bc_lx_mat, self.poisson.coo_dof[:, 0])
        lx_new = np.diff(np.append(lx, lx_0))
        ly = np.matmul(bc_lx_mat, self.poisson.coo_dof[:, 1])
        ly_new = np.diff(np.append(ly, ly_0))
        rx = np.matmul(bc_rx_mat, self.poisson.coo_dof[:, 0])
        rx_new = np.diff(np.append(rx, rx_0))
        ry = np.matmul(bc_rx_mat, self.poisson.coo_dof[:, 1])
        ry_new = np.diff(np.append(ry,  ry_0))
        lr = np.sqrt(lx_new**2 + ly_new**2)
        rr = np.sqrt(rx_new**2 + ry_new**2)
        joints = [torch.tensor(lr).float(), torch.tensor(rr).float()]
        coo_diff = [torch.tensor(lx_0 - lx).float(), torch.tensor(ly_0 - ly).float(),
                    torch.tensor(rx_0 - rx).float(), torch.tensor(ry_0 - ry).float()]

        self.args.load_operators_robot = True
        if self.args.load_operators_robot:
            F00, F01, F10, F11 = np.load(self.args.root_path + '/' + self.args.numpy_path + '/robot/' + 'F' + '.npy')
        else:
            F00, F01, F10, F11 = self.poisson.compute_operators()

        self.F00 = torch.tensor(F00).float().to_sparse()
        self.F01 = torch.tensor(F01).float().to_sparse()
        self.F10 = torch.tensor(F10).float().to_sparse()
        self.F11 = torch.tensor(F11).float().to_sparse()
        self.weight_area = torch.tensor(self.poisson.compute_areas()).float()

        # PyTorch currently has not enough support for sparse matrix (like batch operations)
        # It seems that sparse matrix works fine with SGD but fails for autograd
        # with certain operations. Still looking into this issue.
        # see https://github.com/pytorch/pytorch/issues/9674
        # TODO(Tianju): Do the if selection in optimizer module

        bc_btm_mat = torch.tensor(bc_btm_mat).float().to_sparse()
        bc_lx_mat = torch.tensor(bc_lx_mat).float().to_sparse()
        bc_ly_mat = torch.tensor(bc_ly_mat).float().to_sparse()
        bc_rx_mat = torch.tensor(bc_rx_mat).float().to_sparse()
        bc_ry_mat = torch.tensor(bc_ry_mat).float().to_sparse()
        int_mat = torch.tensor(int_mat).float().to_sparse()

        if self.opt:
            bc_btm_mat = bc_btm_mat.to_dense()
            bc_lx_mat = bc_lx_mat.to_dense()
            bc_ly_mat = bc_ly_mat.to_dense()
            bc_rx_mat = bc_rx_mat.to_dense()
            bc_ry_mat = bc_ry_mat.to_dense()
            int_mat = int_mat.to_dense()
            self.F00 = self.F00.to_dense()
            self.F01 = self.F01.to_dense()
            self.F10 = self.F10.to_dense()
            self.F11 = self.F11.to_dense()

        mat_list = [bc_btm_mat, bc_lx_mat, bc_ly_mat, bc_rx_mat, bc_ry_mat, int_mat]
        self.graph_info = [mat_list, joints, coo_diff, shapes]

 
    def run(self):
        # Subject to change. Raw data generated by some distribution
        raw_data = np.load(self.args.root_path + '/' + self.args.numpy_path + '/' + self.poisson.name 
                              +'/Uniform-30000-' + str(self.poisson.num_dofs) + '.npy')

        left_data = 8*np.transpose(np.matmul(self.bc_l_mat, raw_data.transpose())) 
        right_data = 8*np.transpose(np.matmul(self.bc_r_mat, raw_data.transpose()))

        # # Debug
        # left_data = 0.1*np.ones_like(left_data)
        # right_data = -0.1*np.ones_like(right_data)  
        # left_data[:left_data.shape[0]//2, :] = -0.1
        # right_data[:right_data.shape[0]//2, :] = 0.1
        self.data_X = np.concatenate((left_data, right_data), axis=1)
        self.data_Y = self.data_X
        self.train_loader, self.test_loader = self.shuffle_data()
        self.model = RobotNetwork(self.args, self.graph_info)
        self.model.load_state_dict(torch.load(self.args.root_path + '/' + 
                                              self.args.model_path + '/' + self.poisson.name + '/model_sss'))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        # self.optimizer = optim.LBFGS(self.model.parameters(), lr=1e-4, max_iter=20, history_size=40)

        for epoch in range(self.args.epochs):
            train_loss = self.train(epoch)
            self.fem_test = self.test_X[:1]
            recon_batch = self.model(self.fem_test)
            scalar_field_paraview(self.args, recon_batch[0].data.numpy(), self.poisson, "nn")
            print('\n')
            # torch.save(self.model.state_dict(), self.args.root_path + '/' +
            #            self.args.model_path + '/' + self.poisson.name + '/model_' + str(0))

    def forward_prediction(self, source, model=None, para_data=None):
        """Serves as ground truth computation

        Args:
            source: numpy array (n,)

        """
        source = np.expand_dims(source, axis=0)
        source = torch.tensor(source, dtype=torch.float)
        solver = RobotSolver(self.args, self.graph_info)
        # Load pre-trained network for better convergence
        # model.load_state_dict(torch.load(self.args.root_path + '/' + self.args.model_path + '/robot/solver')) 

        if model is not None:
            solver.reset_parameters_network(source, model)

        if para_data is not None:
            solver.reset_parameters_data(para_data)

        # optimizer = optim.SGD(solver.parameters(), lr=1.*1e-4)
        optimizer = optim.LBFGS(solver.parameters(), lr=1e-1, max_iter=20, history_size=100)
        max_epoch = 100000
        tol = 1e-10
        loss_pre, loss_crt = 0, 0
        for epoch in range(max_epoch):
            def closure():
                optimizer.zero_grad()
                solution = solver(source)
                loss = self.loss_function(source, solution)
                loss.backward()  # Alex: (create_graph=True, retain_graph=True)
                return loss
 
            solution = solver(source)
            loss = self.loss_function(source, solution)
            # print("Optimization for ground truth, loss is", loss.data.numpy())
            assert(not np.isnan(loss.data.numpy()))

            optimizer.step(closure)
            loss_pre = loss_crt
            loss_crt = loss.data.numpy()
            if (loss_pre - loss_crt)**2 < tol:
                break

        # Alex: return torch.autograd.grad(objective(solver(source)), control_params)[0]
        return solution[0].data.numpy(), solver.para.data


    def debug(self):       
        left_data = 0.01*np.ones(self.args.input_size//2)
        right_data = -0.01*np.ones(self.args.input_size//2)
        # left_data[len(left_data)//2:] = -0.1
        # right_data[len(right_data)//2:] = 0.1

        self.model = RobotNetwork(self.args, self.graph_info)
        self.model.load_state_dict(torch.load(self.args.root_path + '/' + self.args.model_path + '/robot/model_sss'))

        source = np.concatenate((left_data, right_data))
        solution = self.adjoint_method(source, model=self.model)
        scalar_field_paraview(self.args, solution, self.poisson, "gt")

        source = torch.tensor(source, dtype=torch.float).unsqueeze(0)
        solution = self.model(source)
        loss = self.loss_function(source, solution)
        print("loss is", loss.data.numpy())
        scalar_field_paraview(self.args, solution.data.numpy().flatten(), self.poisson, "debug")


def get_hessian_inv(J, x):
    H = torch.stack([torch.autograd.grad(J[i], x, retain_graph=True)[0] for i in range(len(J))])
    H_inv = H.inverse()
    return H_inv

if __name__ == "__main__":
    args = arguments.args
    trainer = TrainerRobot(args, True)
    trainer.debug()