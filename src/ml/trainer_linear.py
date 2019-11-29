'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Linear case
'''

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import collections
from .trainer import Trainer
from .models import LinearRegressor
from ..graph.domain import GraphManual
from .. import arguments


class TrainerLinear(Trainer):
    def __init__(self, args):
        super(TrainerLinear, self).__init__(args)
        self.graph = GraphManual(self.args)
        self.args.input_size = self.graph.num_vertices

    def loss_function(self, x_control, x_state):
        # loss function is defined so that PDE is satisfied

        # x_control should be torch tensor with shape (batch, input_size)
        # x_state should be torch tensor with shape (batch, input_size)
        assert(x_control.shape == x_state.shape and len(x_control.shape) == 2)
     
        # to (batch, input_size, 1) for batch matrix-vector multiplication
        x_state = x_state.unsqueeze(2)

        # Sparse representation is ~10 times faster then dense representation in this case
        # dense representation would be lhs = torch.matmul(A, x_state)
        lhs = self.batch_mm(self.A_sp, x_state)

        assert(len(lhs.shape) == 3 and lhs.shape[2] == 1) 

        lhs = lhs.squeeze()
        rhs = x_control
        loss = ((lhs - rhs)**2).sum()
        return loss

    def ground_truth(self):
        gradient_x1_operator = torch.tensor(self.graph.gradient_x1).float()
        gradient_x2_operator = torch.tensor(self.graph.gradient_x2).float()
        boundary_operator = torch.tensor(self.graph.reset_matrix_boundary).float()
        interior_operator = torch.tensor(self.graph.reset_matrix_interior).float()

        tmp = -torch.matmul(gradient_x1_operator, gradient_x1_operator) \
              -torch.matmul(gradient_x2_operator, gradient_x2_operator)
        tmp /= self.graph.num_vertices
        A = torch.matmul(interior_operator, tmp) + boundary_operator

        # If A is ill-conditioned, convergence will be painfully slow
        # Crude check about the condition number
        # print(np.max(A.data.numpy()), np.min(A.data.numpy()))

        A_inv = A.inverse()
        A_sp = A.to_sparse()
        return A, A_sp, A_inv

    def save_progress(self, np_data, L_inf, L_fro, milestone, counter):
        if L_inf < milestone[counter]:
            torch.save(self.model, self.args.root_path + '/' + self.args.model_path + '/linear/model_' + str(counter))
            np.save(self.args.root_path + '/' + self.args.numpy_path + '/linear/error_' + str(counter), np_data)
            counter += 1
        return counter

    def run(self):
        A, A_sp, A_inv = self.ground_truth()
        self.W_true = A_inv.data.numpy()
        self.A_sp = A_sp
        
        # Subject to change. Raw data generated by some distribution
        self.data_X = np.load(self.args.root_path + '/' + self.args.numpy_path + '/linear/' + self.graph.name +
                         '-Uniform-30000-' + str(self.graph.num_vertices) + '.npy')

        self.train_loader, self.test_loader = self.shuffle_data()
        self.model = LinearRegressor(self.args)
        self.model.fc.weight.data = torch.zeros((self.args.input_size, self.args.input_size))

        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        # self.optimizer = optim.LBFGS(self.model.parameters(), lr=1e-2, max_iter=20, history_size=40)
        # self.optimal tuning lr=1e-1, momentum=0.6 for multinomial data
        # self.optimal tuning lr=1e-4, momentum=0.85 for uniform data
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.85)

        milestone = [ 2**(-i) for i in range(-1, 11) ]
        counter = 0
        np_data = np.zeros((2, self.args.epochs))
        for epoch in range(self.args.epochs):
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            W_trained = self.model.fc.weight.data.numpy()
            L_inf = np.max(np.abs(W_trained - self.W_true))
            L_fro = np.linalg.norm(np.abs(W_trained - self.W_true))
            print('====> L_inf norm for matrix error is {}'.format(L_inf))
            print('====> L_fro norm for matrix error is {}'.format(L_fro))
            print('\n\n')
            np_data[:, epoch] = [L_inf, L_fro]
            counter = self.save_progress(np_data, L_inf, L_fro, milestone, counter)
            if L_inf < 1e-4:
                break

    def debug(self):
        model_path = self.args.root_path + '/linear/' + self.args.model_path + '/model'
        self.model =  torch.load(model_path)
        W_trained = self.model.fc.weight.data.numpy()
        print(W_trained)
        print('\n\n\n')
        print(self.W_true.data.numpy())
        print(np.max(np.abs(W_trained - self.W_true)))


if __name__ == "__main__":
    args = arguments.args
    trainer = TrainerLinear(args)
    trainer.run()