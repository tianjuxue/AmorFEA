import numpy as np
import math
from .. import arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm


# Fake data generated using GP
class GaussianProcess(object):
    def __init__(self, args):
        self.args = args

    def generate_input_data(self, n_samps=3000):
        N = 31
        L0 = 1.
        material_distribution = np.zeros((N, N))
        gp_field_index =  np.zeros((2, N, N))
        M = N*N
        kernel_matrix = np.zeros((M, M))
        mean_vector = np.zeros(M)

        for i in range(N):
            for j in range(N):
                gp_field_index[0][i][j], gp_field_index[1][i][j] = self.pixel_to_cartesian(i, j, N, L0)

        for i in range(N):
            for j in range(N):
                row = i*N + j
                mean_vector[row] = mean(gp_field_index[:, i, j])
                print("row is", row)
                for k in range(N):
                    for l in range(N):
                        column = k*N + l
                        kernel_matrix[row][column] = RBF_kernel(gp_field_index[:, i, j], gp_field_index[:, k, l])

        samples = np.random.multivariate_normal(mean_vector, kernel_matrix, n_samps)
        samples = samples.reshape(n_samps, N, N)

        return samples

    def pixel_to_cartesian(self, i, j, N, L0):
        pixel_len = L0/N
        index_x = j - N/2 if j < N/2 else j - N/2 + 1
        index_y = N/2 - i if i < N/2 else N/2 - i - 1
        x = index_x*pixel_len - 0.5*pixel_len if index_x > 0 else index_x*pixel_len + 0.5*pixel_len
        y = index_y*pixel_len - 0.5*pixel_len if index_y > 0 else index_y*pixel_len + 0.5*pixel_len
        return x, y


''' Helper functions'''
def polar(x, y):
    r = (x ** 2 + y ** 2) ** .5
    theta = math.atan2(y, x)
    return r, theta

def RBF_kernel(x1, x2):
    sigma = 1
    l = 0.5
    k = sigma**2 * np.exp( -np.linalg.norm(x1 - x2)**2 / (2*l**2) )
    return k
 
def mean(x):
    return 0

def visualize(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(0, 1, 31)
    Y = np.linspace(0, 1, 31)
    X, Y = np.meshgrid(X, Y)
    Z = data[0]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # plt.axis('off')
    plt.show()

if __name__ == '__main__':
    args = arguments.args
    gp = GaussianProcess(args)
    data = gp.generate_input_data()
    data = np.expand_dims(data, axis=1)
    np.save(args.root_path + '/' + args.numpy_path + '/GP-3000-1-31-31.npy', data)