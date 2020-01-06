import numpy as np
import torch
import scipy.optimize as opt
from .. import arguments
from ..graph.visualization import scalar_field_paraview


if __name__ == '__main__':
    args = arguments.args