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

    def batch_mm(self, matrix, vector_batch):
        batch_size = vector_batch.shape[0]
        # Stack the vector batch into columns. (b, n, 1) -> (n, b)
        vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)

        # A matrix-matrix product is a batched matrix-vector product of the columns.
        # And then reverse the reshaping. (m, b) -> (b, m, 1)
        return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1, 1)

    def shuffle_data(self):
        n_samps = len(self.data_X)
        n_train = int(self.args.train_portion*n_samps)
        inds = np.random.permutation(n_samps)    
        inds_train = inds[:n_train]
        inds_test  = inds[n_train:]
        train_X = self.data_X[inds_train]
        test_X = self.data_X[inds_test]

        train_X = torch.tensor(train_X).float().view(train_X.shape[0], self.args.input_size)
        test_X = torch.tensor(test_X).float().view(test_X.shape[0], self.args.input_size)

        train_data = TensorDataset(train_X)
        test_data = TensorDataset(test_X)
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

        # if test_loss < -3.5:
        #     scalar_field_3D(recon_batch[0].data.numpy(), self.graph)
        #     plt.show()

        return test_loss