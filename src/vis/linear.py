import numpy as np
import math
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from ..opt.optimizer_robot import heart_shape
from .. import arguments

def plot_L(args):
    path_a = args.root_path + '/' + args.numpy_path + '/linear/L_inf_a.npy' 
    path_s = args.root_path + '/' + args.numpy_path + '/linear/L_inf_s.npy' 
    L_inf_a = np.load(path_a)[:-1]
    L_inf_s = np.load(path_s)[:-1]  
    fig = plt.figure()
    ax = fig.gca()
    epoch = np.arange(0, len(L_inf_a), 1)
    ax.plot(epoch, L_inf_a, linestyle='--', marker='o', color='red', label='AFEM')
    ax.plot(epoch, L_inf_s, linestyle='--', marker='o', color='blue', label='Supervised Training')
    # ax.set_yscale('log')
    ax.legend(loc='upper right', prop={'size': 12})
    ax.tick_params(labelsize=14)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


if __name__ == '__main__':
    args = arguments.args
    plot_L(args)
    plt.show()