import fenics as fa
import numpy as np
from .. import arguments
from .poisson import Poisson


if __name__ == '__main__':
    args = arguments.args
    pde = Poisson(args)
    u = pde.solve_problem_weak_form(boundary_fn=fa.Constant(0))
    file = fa.File(args.root_path + '/' + args.solutions_path + '/u.pvd')
    u.rename('u', 'u')
    file << u