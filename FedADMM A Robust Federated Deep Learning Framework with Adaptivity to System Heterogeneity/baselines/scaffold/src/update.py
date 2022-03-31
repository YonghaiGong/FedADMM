import torch
import copy
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.criterion = nn.CrossEntropyLoss()



    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader

    def update_weights(self, model, global_round, alpha, alpha_server, theta):
        local_sum = {}
        control_update = {}
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        global_current = copy.deepcopy(theta)
        model.load_state_dict(theta)
        alpha_temp = copy.deepcopy(alpha)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        param.grad = param.grad + alpha_server[name] - alpha[name]
                optimizer.step()

                if self.args.verbose and (iter % 10 == 0) and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        weights = model.state_dict()
        for key in alpha.keys():
            local_sum[key] = weights[key] - global_current[key]
        for key in alpha_temp.keys():
            a = 1/ (self.args.local_num * self.args.lr) 
            alpha_temp[key] = alpha[key] - alpha_server[key] - local_sum[key] * a
        for key in alpha.keys():
            control_update[key] = alpha_temp[key] - alpha[key]
            alpha[key] = alpha_temp[key]
        
        return sum(epoch_loss) / len(epoch_loss), local_sum, control_update, model, alpha

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(args.device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
