'''
FEDGKD
'''

import torch
import wandb
import logging
import json
import torch.nn.functional as F
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process


class LogitTracker():
    def __init__(self, unique_labels=10, device=torch.device('cpu')):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.device = device
        self.label_counts = torch.ones(unique_labels).cpu()  # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels, unique_labels)).cpu()

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1)).cpu()
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels)).cpu()
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_

    def avg(self):
        res = self.logit_sums / self.label_counts.float().unsqueeze(1)
        return res


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=False, projection=False)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.KL_loss = torch.nn.KLDivLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)
        self.alpha = args.mu
        self.logit_tracker = LogitTracker(10, self.device)
        self.global_logits = None

    def run(self, received_info):
        client_results = []
        logging.info('{} {}'.format(len(self.client_map), self.round))
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info['global'])
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]
            self.global_logits = torch.clone(received_info['avg_logits']).to(self.device) if received_info['avg_logits'] is not None else None
            self.logit_tracker = received_info['prev'][client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc,
                 'client_index': self.client_index, 'user_logits': self.logit_tracker})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        self.round += 1
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                #####
                out = self.model(x)
                self.logit_tracker.update(out.detach().cpu(), target.detach().cpu())
                loss1 = self.criterion(out, target)
                if self.global_logits != None:
                    target_p = F.softmax(self.global_logits[target, :], dim=1)
                    loss2 = self.KL_loss(out, target_p)

                else:
                    loss2 = 0

                loss = loss1 + loss2 * self.alpha
                #####
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss1.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()
        # self.prev_model.load_state_dict(weights)
        return weights


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=False, projection=False)
        self.logits = {x: LogitTracker(self.num_classes) for x in range(self.args.client_number)}
        wandb.watch(self.model)

    def run(self, received_info):
        server_outputs = self.operations(received_info)

        user_logits = [c['user_logits'] for c in received_info]
        self.user_logits = [torch.zeros(user_logits[0].unique_labels)]

        for i in range(len(user_logits)):
            for j in range(len(user_logits)):
                if i != j:
                    self.user_logits[i] += user_logits[j].avg() / (len(user_logits) - 1)

        acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc

        for x in received_info:
            self.logits[x['client_index']] = x['user_logits']

        server_outputs = [{'global': g, 'avg_logits': self.user_logits[i], 'prev': self.logits} for i, g in enumerate(server_outputs)]
        return server_outputs

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [{'global': self.model.cpu().state_dict(), 'avg_logits': None, 'prev': self.logits} for x in
                range(self.args.thread_number)]