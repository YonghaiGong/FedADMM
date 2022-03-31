import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default = 200,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default = 0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--epochs', type=int, default = 100,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default = 10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default = 50,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default = 0.1,
                        help='learning rate')
    parser.add_argument('--rho', type=float, default = 0.1,
                        help='hpy')
    parser.add_argument('--model', type=str, default='Model1', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--file_name', type=str, default='prox',
                        help='file name.')
    parser.add_argument('--seed', type=int, default = 2022,
                        help="random seed")
    parser.add_argument('--fixed', type=int, default = 1,
                        help="fixed local epoch number, 1 for fixed")
    parser.add_argument('--threshold', type=float, default = 1.0,
                        help="client threshold to random choose lcoal epochs")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    args = parser.parse_args()
    return args
