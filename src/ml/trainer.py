from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
from .generator import cube_material_batch
from .models import VAE_FC, VAE_CNN
from .. import arguments
from ..utils.constants import *
from sklearn.decomposition import PCA


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, recon_x, y, y_hat, mu, logvar, joint_traing=False):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE_criteria = nn.MSELoss(reduction='sum')
    MSE = MSE_criteria(y, y_hat)
    return BCE + KLD + MSE if joint_traing else BCE + KLD 

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        optimizer.zero_grad()
        recon_batch, prediction, mu, logvar = model(data)

        loss = loss_function(data, recon_batch, target, prediction, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.float()
            recon_batch, prediction, mu, logvar = model(data)
            test_loss += loss_function(data, recon_batch, target, prediction, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, args.input_size, args.input_size)[:n]])
                save_image(comparison,
                           args.root_path + '/' + args.images_path + '/reconstruction_' + str(epoch) + '.png', 
                           nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def plot_E(target, prediction):
    target = target.data.numpy()
    prediction = prediction.data.numpy()
    plt.figure()
    x = np.linspace(0, 1, len(target))
    plt.plot(x, target, '-*', label='true')
    plt.plot(x, prediction, '-*', label='prediction')
    # plt.xlabel("Relative Young's modulus")   
    # plt.ylabel("Poisson's ratio")
    plt.legend(loc="lower right")

def plot_latent(z, E):
    E = E.reshape(-1)
    # fig = plt.figure(figsize=(15, 4))
    fig = plt.figure()
    cmap = plt.get_cmap('viridis', 10)

    for i in range(1):
        # plt.subplot(1, 1, i+1)
        im = plt.scatter(z[:, 0], z[:, 1], c=E, 
                         cmap=cmap, 
                         vmin=0, vmax=1, 
                         marker='o', s=10)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)

    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, fig.add_axes([0.82, 0.13, 0.02, 0.74]), ticks=np.linspace(0, 1, 11))

def threshold(sample):
    t = 0.5
    sample[np.where(sample < t)] = 0
    sample[np.where(sample >= t)] = 1
    return sample

def filter_cube(sample):
    # assert(len(sample.shape) == 4)
    sample_batch = sample.shape[0]
    input_size = sample.shape[-1]
    sample = sample.data.numpy()
    sample = sample.reshape(sample_batch, input_size, input_size)
    sample = threshold(sample)
    sample = cube_material_batch(sample)
    sample = sample.reshape(sample_batch, 1, input_size*2, input_size*2)
    sample = torch.tensor(sample)
    return sample

def normalize_labels(data_Y):
    min_Y = np.repeat(np.expand_dims(np.min(data_Y, axis=0), axis=0), len(data_Y), axis=0)
    max_Y = np.repeat(np.expand_dims(np.max(data_Y, axis=0), axis=0), len(data_Y), axis=0)
    range_Y = max_Y - min_Y
    data_Y_normalized = (data_Y - min_Y) / range_Y   
    return data_Y_normalized

def shuffle_data(data_X, data_Y):
    n_samps = len(data_X)
    n_train = int((1 - 0.2) * n_samps)
    inds = np.random.permutation(n_samps)    
    inds_train = inds[:n_train]
    inds_test  = inds[n_train:]
    train_X = data_X[inds_train]
    train_Y = data_Y[inds_train]
    test_X = data_X[inds_test]
    test_Y = data_Y[inds_test]
    return train_X, train_Y, test_X, test_Y

def generate_vae_images():
    sample = torch.randn(100, args.hidden_size)
    sample = model.decode(sample)
    sample = filter_cube(sample)
    sample = sample.data.numpy()
    sample = sample.reshape(sample.shape[0], args.input_size*2, args.input_size*2)
    np.save(args.root_path + '/' + args.numpy_path + '/VAE-100-56-56.npy', sample)


if __name__ == "__main__":

    args = arguments.args

    analysis = False

    torch.manual_seed(args.seed)
    data_X = np.load(args.root_path + '/' + args.numpy_path + '/GP-3000-1-28-28-RBF.npy')
    data_Y = np.load(args.root_path + '/' + args.numpy_path + '/GP-3000-mp-RBF.npy')
    args.input_size = data_X.shape[2]
    data_Y = np.zeros((30000, 3))

    data_Y = normalize_labels(data_Y)

    train_X, train_Y, test_X, test_Y = shuffle_data(data_X, data_Y)
    train_X, \
    train_Y, \
    test_X, \
    test_Y = \
    torch.tensor(train_X).float(), \
    torch.tensor(train_Y).float(), \
    torch.tensor(test_X).float(), \
    torch.tensor(test_Y).float() 

    if analysis:

        model_path = args.root_path + '/' + args.model_path + '/model'
        model =  torch.load(model_path)

        # plot_E(target, prediction)
        z, _, _ = model.encode(train_X)
        y_hat = model.predict(z)
        y = train_Y

        z = z.data.numpy()    
        y = y.data.numpy()
        y_hat = y_hat.data.numpy()

        pca = PCA(n_components=2)
        z_compressed = pca.fit_transform(z)
 
        plot_latent(z_compressed, y[:, :1])
        plot_latent(z_compressed, y[:, 1:2])
        plot_latent(z_compressed, y[:, 2:3])

        plt.show()

    else:

        train_data = TensorDataset(train_X, train_Y)
        test_data = TensorDataset(test_X, test_Y)
        train_loader =  DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        model = VAE_FC(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            with torch.no_grad():
                sample = torch.randn(16, args.hidden_size)
                sample = model.decode(sample)
                sample_np = sample.data.numpy()
                sample = filter_cube(sample)
                save_image(sample, 
                           args.root_path + '/' + args.images_path + '/sample_' + str(epoch) + '.png', 
                           padding=10, 
                           nrow=4)
                sample_np = threshold(sample_np)
                np.save(args.root_path + '/' + args.others_path + \
                        '/Strokes-generated-1-28-28-epoch' + str(epoch) + '.npy',
                        sample_np)

        torch.save(model, args.root_path + '/' + args.model_path + '/model')
