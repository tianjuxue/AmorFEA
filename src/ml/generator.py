import numpy as np
import math
from .. import arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from ..graph.visualization import scalar_field_paraview
from ..pde.poisson_linear import PoissonLinear
from ..pde.poisson_dolfin import PoissonDolfin
from ..pde.poisson_robot import PoissonRobot


def generate_gaussian_samples(args, pde, num_samps, case_flag):

    if case_flag == 1:
        def RBF_kernel(x1, x2):
            sigma = 100
            l = 0.1
            k = sigma**2 * np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * l**2))
            return k
    elif case_flag == 2:
        def RBF_kernel(x1, x2):
            sigma = 1
            l1 = 0.001
            l2 = 5
            d1 = (x1[0] - x2[0])**2
            d2 = (x1[1] - x2[1])**2
            k = sigma**2 * np.exp(-d1 / (2 * l1**2) - d2 / (2 * l2**2))
            return k
    else:
        raise NotImplementedError()

    def mean(x):
        return 0.

    M = pde.num_dofs
    coo = pde.coo_dof
    kernel_matrix = np.zeros((M, M))
    mean_vector = np.zeros(M)

    for i in range(M):
        mean_vector[i] = mean(coo[i])
        for j in range(M):
            kernel_matrix[i][j] = RBF_kernel(coo[i], coo[j])

    samples = np.random.multivariate_normal(
        mean_vector, kernel_matrix, num_samps)
    save_generated_data(args, pde, 'Gaussian', samples)
    return samples


def generate_uniform_samples(args, pde, num_samps, case_flag):
    M = pde.num_dofs
    samples = np.random.uniform(-1, 1, (num_samps, M))
    save_generated_data(args, pde, 'Uniform', samples)
    return samples


def generate_multinomial_samples(args, pde, num_samps):
    M = pde.num_dofs
    samples = np.random.multinomial(1, [1 / M] * M, size=num_samps)
    save_generated_data(args, pde, 'Multi', samples)
    return samples


def save_generated_data(args, pde, distribution, samples):
    np.save(args.root_path + '/' + args.numpy_path + '/' + pde.name + '/' + distribution +
            '-' + str(samples.shape[0]) + '-' + str(pde.num_dofs) + '.npy', samples)


if __name__ == '__main__':
    args = arguments.args
    case_flag = 1
    if case_flag == 0:
        poisson_linear = PoissonLinear(args)
        samples = generate_uniform_samples(
            args, poisson_linear, 10000, case_flag)
    elif case_flag == 1:
        poisson_dolfin = PoissonDolfin(args)
        samples = generate_gaussian_samples(
            args, poisson_dolfin, 30000, case_flag)
    else:
        poisson_robot = PoissonRobot(args)
        samples = generate_uniform_samples(
            args, poisson_robot, 30000, case_flag)
