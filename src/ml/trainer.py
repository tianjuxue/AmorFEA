'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Base class definition
'''
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
        torch.manual_seed(self.args.seed)

    def shuffle_data(self):
        n_samps = len(self.data_X)
        n_train = int(self.args.train_portion*n_samps)
        inds = np.random.permutation(n_samps)    
        inds_train = inds[:n_train]
        inds_test  = inds[n_train:]
        train_X = self.data_X[inds_train]
        test_X = self.data_X[inds_test]

        self.train_X = torch.tensor(train_X).float().view(train_X.shape[0], self.args.input_size)
        self.test_X = torch.tensor(test_X).float().view(test_X.shape[0], self.args.input_size)

        train_data = TensorDataset(self.train_X)
        test_data = TensorDataset(self.test_X)
        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True)

        return train_loader, test_loader

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):

            # In case we need second order optimizer, a closure has to be defined
            def closure():
                self.optimizer.zero_grad()
                recon_batch = self.model(data)
                loss = self.loss_function(data, recon_batch)
                loss.backward()
                return loss

            data = data[0].float()
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(data, recon_batch)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step(closure)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        train_loss /= len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
        return train_loss

    def test_by_loss(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data[0].float()
                recon_batch = self.model(data)
                test_loss += self.loss_function(data, recon_batch).item()
        test_loss /= len(self.test_loader.dataset)
        print('====> Epoch: {} Test set loss: {:.6f}'.format(epoch, test_loss))
        return test_loss

    def FEM_evaluation(self):
        num_fem_test = 100
        self.fem_test = self.test_X[:num_fem_test]
        fem_solution = []
        for i in range(num_fem_test):
            self.poisson.set_control_variable(self.fem_test[i])
            u = self.poisson.solve_problem_variational_form()
            dof_data = u.vector()[:]
            fem_solution.append(dof_data)
        self.fem_solution = torch.Tensor(fem_solution).float()

    def test_by_FEM(self, epoch):
        self.model.eval()
        recon_batch = self.model(self.fem_test)
        error = recon_batch - self.fem_solution
        tmp = batch_mat_vec(self.B_sp, error)
        tmp = tmp*error
        mean_L2_error = tmp.sum(dim=1).sqrt().mean()

        print('====> Mean L2 error: {:.8f}'.format(mean_L2_error))

        index = epoch%self.fem_test.shape[0]
        scalar_field_paraview(self.args, self.fem_test[index].data.numpy(), self.poisson, "source")
        scalar_field_paraview(self.args, self.fem_solution[index].data.numpy(), self.poisson, "fem")
        scalar_field_paraview(self.args, recon_batch[index].data.numpy(), self.poisson, "nn")

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
    x = torch.stack([torch.mm(sparse_matrix, matrix_batch[i].float()) for i in range(batch_size)])
    # x = torch.matmul(sparse_matrix.to_dense(), matrix_batch)
    return x

def normalize_adj(A):
    size = A.shape[0]
    A = A + np.identity(size)
    D = np.array(A.sum(1))
    D = np.diag(D**(-0.5))
    A_normalized = np.matmul(np.matmul(D, A), D)
    A_normalized = torch.tensor(A_normalized).float()
    A_sp = A_normalized.to_sparse()
    return A_sp
    # return torch.tensor(A).float().to_sparse() 

def boundary_flag_matrix(boundary_flag):
    # something like [0,0,1,1,0] to
    # [[0,0,1,0,0], [0,0,0,1,0]]
    bc_mat = []
    for i, number in enumerate(boundary_flag):
        if number == 1:
            row = np.zeros(len(boundary_flag))
            row[i] = 1
            bc_mat.append(row)
    return np.array(bc_mat)
