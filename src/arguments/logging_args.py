"""Args defining output and logging"""


def add_args(parser):
    parser.add_argument(
        '--verbose',
        help='Verbose for debug',
        action='store_true',
        default=False)
