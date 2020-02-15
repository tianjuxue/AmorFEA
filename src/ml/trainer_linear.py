import torch
from torch import optim
import numpy as np
from .trainer import Trainer, batch_mat_vec, normalize_adj
from .models import LinearRegressor
from ..pde.poisson_linear import PoissonLinear
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class TrainerLinear(Trainer):

    def __init__(self, args):
        super(TrainerLinear, self).__init__(args)
        self.poisson = PoissonLinear(self.args)
        self.initialization()

    def loss_function(self, x_control, x_state, y_state):
        if self.args.supvervised_flag:
            return self.loss_function_supervised(x_state, y_state)
        else:
            return self.loss_function_amortized(x_control, x_state)

    def loss_function_amortized(self, x_control, x_state):
        # x_control should be torch tensor with shape (batch, input_size)
        # x_state should be torch tensor with shape (batch, input_size)
        assert(x_control.shape == x_state.shape and len(x_control.shape) == 2)

        tmp1 = batch_mat_vec(self.A_sp, x_state)
        tmp1 = 0.5 * tmp1 * x_state
        tmp2 = batch_mat_vec(self.B_sp, x_control)
        tmp2 = tmp2 * x_state
        loss = tmp1.sum() - tmp2.sum()
        return loss

    def loss_function_supervised(self, x_state, y_state):
        diff = (x_state - y_state)
        tmp = batch_mat_vec(self.B_sp, diff)
        loss = diff * tmp
        loss = loss.sum()
        return loss

    def initialization(self):
        self.args.supvervised_flag = True

        self.data_X = np.load(self.args.root_path + '/' + self.args.numpy_path + '/' + self.poisson.name +
                              '/Uniform-10000-' + str(self.poisson.num_dofs) + '.npy')

        self.args.load_fem_data = True
        if self.args.load_fem_data:
            self.data_Y = np.load(self.args.root_path + '/' + self.args.numpy_path + '/' + self.poisson.name +
                                  '/fem_solution.npy')
        else:
            self.FEM_evaluation_all()

        self.args.input_size = self.data_X.shape[1]
        self.train_loader, self.test_loader = self.shuffle_data()

        self.A_np, self.B_np, self.A_np_modified = self.poisson.compute_operators()
        A = torch.tensor(self.A_np).float()
        B = torch.tensor(self.B_np).float()
        self.A_sp = A.to_sparse()
        self.B_sp = B.to_sparse()
        self.A_inv = np.linalg.inv(self.A_np_modified)

        # Can be more general
        # Fixed bc for now
        bc_flag = torch.tensor(self.poisson.boundary_flags_list[0]).float()
        bc_value = 0. * bc_flag
        interior_flag = torch.ones(self.poisson.num_dofs) - bc_flag
        adjacency_matrix = self.poisson.get_adjacency_matrix()
        A_normalized = normalize_adj(adjacency_matrix)
        self.graph_info = [bc_value, interior_flag, A_normalized, self.B_sp]

        self.reset_matrix_boundary = np.diag(self.poisson.boundary_flags)
        self.reset_matrix_interior = np.identity(
            self.poisson.num_dofs) - self.reset_matrix_boundary

        self.FEM_evaluation()

    def test_by_loss(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data_x = data[0].float()
                data_y = data[1].float()
                recon_batch = self.model(data_x)
                test_loss += self.loss_function(data_x,
                                                recon_batch, data_y).item()
        test_loss /= len(self.test_loader.dataset)
        print('====> Epoch: {} Test set loss: {:.6f}'.format(epoch, test_loss))

    def test_by_W(self, epoch):
        self.model.eval()
        W_trained = self.model.fcc.weight.data.numpy()

        tmp = np.matmul(self.reset_matrix_interior, W_trained)
        tmp = np.matmul(tmp, self.reset_matrix_interior)
        Q_trained = np.matmul(tmp, self.B_np)

        tmp = np.matmul(self.A_inv, self.reset_matrix_interior)
        Q_true = np.matmul(tmp, self.B_np)

        L_inf = np.max(np.abs(Q_trained - Q_true))
        L_inf_norm = np.max(np.abs(Q_true))
        L_fro = np.linalg.norm(np.abs(Q_trained - Q_true))

        print('====> L_inf norm for matrix error is {}'.format(L_inf / L_inf_norm))
        return L_inf / L_inf_norm

    def run(self):
        self.model = LinearRegressor(self.args, self.graph_info)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        L_inf_list = []
        for epoch in range(self.args.epochs):
            train_loss = self.train(epoch)
            test_loss = self.test_by_loss(epoch)
            mean_L2_error = self.test_by_FEM(epoch)
            L_inf = self.test_by_W(epoch)
            L_inf_list.append(L_inf)
            if epoch > 10:
                np.save(self.args.root_path + '/' + self.args.numpy_path + '/' + self.poisson.name +
                        '/L_inf.npy', np.asarray(L_inf_list))
                exit()

            print('\n\n')
            if False:
                torch.save(self.model, self.args.root_path + '/' +
                           self.args.model_path + '/linear/model_' + str(0))

    def debug(self):
        source = torch.ones(self.poisson.num_dofs).unsqueeze(0)
        solution = self.model(source)
        scalar_field_paraview(
            self.args, solution.data.numpy().flatten(), self.poisson, "ok")


if __name__ == "__main__":
    args = arguments.args
    trainer = TrainerLinear(args)
    trainer.run()
