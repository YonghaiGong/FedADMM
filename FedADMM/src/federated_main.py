import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import Model1, Model2
from utils import get_dataset, average_weights, exp_details, setup_seed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


if __name__ == '__main__':
    start_time = time.time()

    path_project = os.path.abspath('.')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    setup_seed(args.seed)
    print('random seed =', args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, user_groups = get_dataset(args)

    local_model, model = [], []

    if args.model == 'Model1':
        global_model = Model1(args=args)
    elif args.model == 'Model2':
        global_model = Model2(args=args)
    else:
        exit('Error: unrecognized model')

    global_model.to(args.device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()
    theta = copy.deepcopy(global_weights)

    for idx in range(args.num_users):
        local_model.append(LocalUpdate(args=args, model=global_model, \
            dataset=train_dataset, idxs=user_groups[idx], logger=logger))
         
    test_acc, test_loss = [], []
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)): 

        local_losses, local_sum = [], []   
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            loss, lsum = local_model[idx].update_weights(global_round=epoch, theta=copy.deepcopy(theta))

            local_losses.append(copy.deepcopy(loss))
            local_sum.append(copy.deepcopy(lsum))

        update_msg = average_weights(local_sum)

        if epoch >= args.target_round:
        	args.eta = args.eta_2
        for key in theta.keys():
            theta[key] = update_msg[key] * args.eta + theta[key]

        global_model.load_state_dict(theta)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        test_acc_1, test_loss_1 = test_inference(args, global_model, test_dataset)
        print('\ntest accuracy:{:.2f}%\n'.format(100*test_acc_1))
        test_acc.append(test_acc_1)
        test_loss.append(test_loss_1)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[-1]))

    file_name = '../save/{}_{}.pkl'.format(args.file_name, args.seed)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, test_acc], f)


