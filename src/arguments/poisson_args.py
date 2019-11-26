"""metamaterial.py arguments"""


def add_args(parser):
    parser.add_argument('--L0', type=float, default=1e-3/31)
    parser.add_argument('--n_cells', type=int, default=31) 