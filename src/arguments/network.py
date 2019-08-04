"""Args defining output and networks"""


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--input_size', type=int, default=28, 
                        help='input size')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')