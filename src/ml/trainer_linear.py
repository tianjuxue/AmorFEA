'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Linear case
'''

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import collections
from .trainer import Trainer, batch_mat_vec, normalize_adj
from .models import LinearRegressor
from ..pde.poisson_square import PoissonSquare
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class TrainerLinear(Trainer):
    def __init__(self, args):
        super(TrainerLinear, self).__init__(args)
        self.poisson = PoissonSquare(self.args)
        self.initialization()

    def loss_function(self, x_control, x_state):
        # loss function is defined so that PDE is satisfied

        # x_control should be torch tensor with shape (batch, input_size)
        # x_state should be torch tensor with shape (batch, input_size)
        assert(x_control.shape == x_state.shape and len(x_control.shape) == 2)
     
        tmp1 = batch_mat_vec(self.A_sp, x_state)
        tmp1 = 0.5*tmp1*x_state
        tmp2 = batch_mat_vec(self.B_sp, x_control)
        tmp2 = tmp2*x_state
        loss = tmp1.sum() - tmp2.sum()
        return loss

    def initialization(self):

        # Subject to change. Raw data generated by some distribution
        self.data_X = np.load(self.args.root_path + '/' + self.args.numpy_path + '/linear/' + 
                         self.poisson.name + '-Uniform-30000-' + str(self.poisson.num_dofs) + '.npy')
        self.args.input_size = self.data_X.shape[1]
        self.train_loader, self.test_loader = self.shuffle_data()

        A_np, B_np, A_np_modified = self.poisson.compute_operators()
        A = torch.tensor(A_np).float()
        B = torch.tensor(B_np).float()
        self.A_sp = A.to_sparse()
        self.B_sp = B.to_sparse()
        self.W_true = np.linalg.inv(A_np_modified)

        # Can be much more general
        # Fixed bc for now
        bc_flag = torch.tensor(self.poisson.boundary_flags_list[0]).float()
        bc_value = 0.*bc_flag
        interior_flag = torch.ones(self.poisson.num_dofs) - bc_flag
        adjacency_matrix = self.poisson.get_adjacency_matrix()
        A_normalized = normalize_adj(adjacency_matrix)
        self.graph_info = [bc_value, interior_flag, A_normalized, self.B_sp]

        self.reset_matrix_boundary = np.diag(self.poisson.boundary_flags)
        self.reset_matrix_interior = np.identity(self.poisson.num_dofs) - self.reset_matrix_boundary

        self.FEM_evaluation()  

    def run(self):

        self.model = LinearRegressor(self.args, self.graph_info)
        # self.model.fc.weight.data = torch.zeros((self.args.input_size, self.args.input_size))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer = optim.LBFGS(self.model.parameters(), lr=1e-2, max_iter=20, history_size=40)
        # self.optimal tuning lr=1e-1, momentum=0.6 for multinomial data
        # self.optimal tuning lr=1e-4, momentum=0.85 for uniform data
        # self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.85)

        for epoch in range(self.args.epochs):
            train_loss = self.train(epoch)
            test_loss = self.test_by_loss(epoch)
            mean_L2_error = self.test_by_FEM(epoch)
            # L_inf, L_fro = self.test_by_W(epoch)
            print('\n\n')

            if mean_L2_error < 1e-4:
                self.debug()
                exit()

            # if True:
            #     torch.save(self.model, self.args.root_path + '/' + self.args.model_path + '/linear/model_' + str(0))

    def test_by_W(self, epoch):
        self.model.eval()
        W_trained = self.model.fc.weight.data.numpy()
        W_trained = np.matmul(self.reset_matrix_interior, W_trained) + self.reset_matrix_boundary
        L_inf = np.max(np.abs(W_trained - self.W_true))
        L_fro = np.linalg.norm(np.abs(W_trained - self.W_true))
        print('====> L_inf norm for matrix error is {}'.format(L_inf))
        print('====> L_fro norm for matrix error is {}'.format(L_fro))
        return L_inf, L_fro

    def debug(self):
        source = torch.ones(self.poisson.num_dofs).unsqueeze(0)
        solution = self.model(source)
        scalar_field_paraview(self.args, solution.data.numpy().flatten(), self.poisson, "ok")


if __name__ == "__main__":
    args = arguments.args
    trainer = TrainerLinear(args)
    trainer.run()