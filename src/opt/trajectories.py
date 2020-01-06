import matplotlib.pyplot as plt
import numpy as np
from .. import arguments


def x_para(t):
    return 16*np.sin(t)**3

def y_para(t):
    return 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t) - 5

def heart_shape():
    vertical_dist = 2
    norm_factor = vertical_dist / (y_para(0) - y_para(np.pi))
    t = np.linspace(0, 2*np.pi, 31)
    x = norm_factor*x_para(t)
    y = norm_factor*y_para(t)
    return np.asarray([x, y])

def plot_hs():
    x, y = heart_shape()
    fig = plt.figure(0)
    plt.tick_params(labelsize=14)
    # plt.xlabel('xlabel')
    # plt.ylabel('ylabel')
    # plt.legend(loc='upper left')
    plt.plot(x, y)


if __name__ == "__main__":
    args = arguments.args
    plot_hs()
    plt.show()