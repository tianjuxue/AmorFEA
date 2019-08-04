"""metamaterial.py arguments"""


def add_args(parser):
    parser.add_argument('--L0', type=float, default=0.5)
    parser.add_argument('--n_cells', type=int, default=30) 