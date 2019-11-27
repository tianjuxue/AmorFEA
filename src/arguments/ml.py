"""Args defining output and networks"""


def add_args(parser):
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--input_size', type=int, default=16, 
                        help='input size')
    parser.add_argument('--train_portion', type=float, default=0.9, 
                        help='train and test split portion for train')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train')
    # Linear trainer use 512
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate')

