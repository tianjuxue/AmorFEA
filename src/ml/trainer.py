'''Train the neural network to be a powerful PDE solver, where physical laws have been built in
Base class definition
'''
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from .. import arguments
from ..graph.visualization import *


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

    def test(self, epoch):
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


# # '''Helpers'''
def batch_mat_vec(sparse_matrix, vector_batch):
    # sparse_matrix: (k, n)
    # vector_batch: (b, n)
    # want to do batch multiplication and return (b, k)

    # (b, n) -> (n, b)
    matrices = vector_batch.transpose(0, 1)

    # (k, b) -> (b, k)
    return sparse_matrix.mm(matrices).transpose(1, 0)

def batch_mat_mat(sparse_matrix, matrix_batch):
    batch_size = matrix_batch.shape[0]
    x = torch.stack([torch.mm(sparse_matrix, matrix_batch[i].float()) for i in range(batch_size)])
    # x = torch.matmul(sparse_matrix.to_dense(), matrix_batch)
    # x = torch.matmul(sparse_matrix, matrix_batch)
    return x