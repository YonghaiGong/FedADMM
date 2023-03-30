import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_users', type=int, default = 100,
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
    parser.add_argument('--rho', type=float, default = 0.01,
                        help='hpy')
    parser.add_argument('--model', type=str, default='Model2', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--file_name', type=str, default='name',
                        help='file name.')
    parser.add_argument('--seed', type=int, default = 707,
                        help="random seed")
    parser.add_argument('--fixed', type=int, default = 0,
                        help="fixed local epochs, 1 for fixed")
    parser.add_argument('--threshold', type=float, default = 1.0,
                        help="client threshold to random choose lcoal epochs")
    parser.add_argument('--eta', type=float, default = 1,
                        help="learning rate of global model")
    parser.add_argument('--eta_2', type=float, default = 0.5,
                        help="learning rate of global model phase 2")
    parser.add_argument('--target_round', type=int, default = 60,
                        help="the number of target round to change eta")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    args = parser.parse_args()
    return args
