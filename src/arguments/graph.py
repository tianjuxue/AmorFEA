"""Args defining graph"""


def add_args(parser):
    parser.add_argument('--assemble_flag', help='whether to assemble', 
                        action='store_true', default=False)