import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from .. import arguments
from ..graph.visualization import scalar_field_paraview


class Trainer(object):
    """Base class for Trainer.
    """

    def __init__(self, args):
        self.args = args

    def shuffle_data(self):
        n_samps = len(self.data_X)
        n_train = int(self.args.train_portion * n_samps)
        inds = np.random.permutation(n_samps)
        inds_train = inds[:n_train]
        inds_test = inds[n_train:]

        train_X = self.data_X[inds_train]
        test_X = self.data_X[inds_test]

        self.train_X = torch.tensor(train_X, dtype=torch.float)
        self.test_X = torch.tensor(test_X, dtype=torch.float)

        train_Y = self.data_Y[inds_train]
        test_Y = self.data_Y[inds_test]
        self.train_Y = torch.tensor(train_Y, dtype=torch.float)
        self.test_Y = torch.tensor(test_Y, dtype=torch.float)

        train_data = TensorDataset(self.train_X, self.train_Y)
        test_data = TensorDataset(self.test_X, self.test_Y)

        train_loader = DataLoader(
            train_data, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_data, batch_size=self.args.batch_size, shuffle=True)

        return train_loader, test_loader

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            def closure():
                self.optimizer.zero_grad()
                recon_batch = self.model(data_x)
                loss = self.loss_function(data_x, recon_batch, data_y)
                loss.backward()
                return loss

            data_x = data[0].float()
            data_y = data[1].float()
            recon_batch = self.model(data_x)
            loss = self.loss_function(data_x, recon_batch, data_y)
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data_x), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data_x)))

            train_loss += loss.item()
            self.optimizer.step(closure)

        train_loss /= len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        return train_loss

    def FEM(self, source):
        fem_solution = []
        for i in range(len(source)):
            self.poisson.set_control_variable(source[i])
            u = self.poisson.solve_problem_variational_form()
            dof_data = u.vector()[:]
            fem_solution.append(dof_data)
        fem_solution = torch.Tensor(fem_solution).float()
        return fem_solution

    def FEM_evaluation_all(self):
        self.data_Y = self.FEM(self.data_X).data.numpy()
        np.save(self.args.root_path + '/' + self.args.numpy_path + '/' + self.poisson.name +
                '/fem_solution.npy', self.data_Y)

    # TODO(Tianju): Merge this test to normed_L2_error
    def FEM_evaluation(self):
        num_fem_test = 100
        start = 0
        self.fem_test = self.test_X[start:start + num_fem_test]
        self.fem_solution = self.FEM(self.fem_test)

    def test_by_FEM(self, epoch):
        recon_batch = self.model(self.fem_test)
        error = recon_batch - self.fem_solution
        tmp = batch_mat_vec(self.B_sp, error)
        tmp = tmp * error
        L2_error = tmp.sum(dim=1).sqrt()
        mean_L2_error = L2_error.mean()
        print('====> Mean L2 error: {:.8f}'.format(mean_L2_error))
        index = epoch % self.fem_test.shape[0]
        scalar_field_paraview(self.args, self.fem_test[
                              index].data.numpy(), self.poisson, "train_f")
        scalar_field_paraview(self.args, self.fem_solution[
                              index].data.numpy(), self.poisson, "train_fem_u")
        scalar_field_paraview(self.args, recon_batch[
                              index].data.numpy(), self.poisson, "train_nn_u")

        return mean_L2_error


'''Helpers'''


def batch_mat_vec(sparse_matrix, vector_batch):
    """Supports batch matrix-vector multiplication for both sparse matrix and dense matrix.

    Args:
        sparse_matrix: torch tensor (k, n). Can be sparse or dense.
        vector_batch: torch tensor (b, n).

    Returns:
        vector_batch: torch tensor (b, k)

    """

    # (b, n) -> (n, b)
    matrices = vector_batch.transpose(0, 1)

    # (k, b) -> (b, k)
    return sparse_matrix.mm(matrices).transpose(1, 0)


def batch_mat_mat(sparse_matrix, matrix_batch):
    batch_size = matrix_batch.shape[0]
    x = torch.stack([torch.mm(sparse_matrix, matrix_batch[i].float())
                     for i in range(batch_size)])
    # x = torch.matmul(sparse_matrix.to_dense(), matrix_batch)
    return x


def normalize_adj(A):
    size = A.shape[0]
    A = A + np.identity(size)
    D = np.array(A.sum(1))
    D = np.diag(D**(-0.5))
    A_normalized = np.matmul(np.matmul(D, A), D)
    A_normalized = torch.tensor(A_normalized).float()
    return A_normalized.to_sparse()
    # return torch.tensor(A).float().to_sparse()


def boundary_flag_matrix(boundary_flag):
    """something like [0,0,1,1,0] to [[0,0,1,0,0], [0,0,0,1,0]]
    """

    bc_mat = []
    for i, number in enumerate(boundary_flag):
        if number == 1:
            row = np.zeros(len(boundary_flag))
            row[i] = 1
            bc_mat.append(row)
    return np.array(bc_mat)
