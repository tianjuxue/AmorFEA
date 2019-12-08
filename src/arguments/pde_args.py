"""pde.py arguments"""


def add_args(parser):
    parser.add_argument('--nonlinear_solver', default='newton', type=str,
        help='Nonlinear solver: newton or snes')    
    parser.add_argument('--relaxation_parameter', default=0.1, type=float, 
        help='relaxation parameter for Newton')
    parser.add_argument('--linear_solver', default='petsc', type=str,
        help='Newton linear solver')
    parser.add_argument('--max_newton_iter', default=300, type=int,
        help='Newton maximum iters')
    parser.add_argument('--rel_tol', default=1e-5, type=float,
        help='relative tolerance')
    parser.add_argument('--abs_tol', default=1e-5, type=float,
        help='absolute tolerance')
    parser.add_argument('--L0', type=float, default=0.1)
    parser.add_argument('--n_cells', type=int, default=5) 