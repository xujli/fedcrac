'''
FedTrip
'''

import torch
import wandb
import logging
from copy import deepcopy
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.prev_model = self.model_type(self.num_classes).to(self.device)
        # self.prev_model.load_state_dict(self.model.state_dict())
        self.global_model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)
        self.alpha = args.mu
        self.epochs = args.epochs

    def run(self, received_info):
        client_results = []
        logging.info(len(self.client_map))
        self.global_model.load_state_dict(received_info['global'])
        for client_idx in self.client_map[self.round]:
            self.last_model = deepcopy(received_info['prev'][client_idx])
            self.gap = max(self.round - received_info['rounds'][client_idx] + 1, 1)
            self.load_client_state_dict(received_info['global'])
            self.train_dataloader = self.train_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]

            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size
            weights = self.train()
            acc = self.test()
            client_results.append(
                {'weights': weights, 'num_samples': num_samples, 'acc': acc, 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        self.round += 1
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.global_model.to(self.device)
        self.model.train()
        epoch_loss = []
        self.global_model_params = deepcopy(self.global_model.state_dict())

        cnt = 0
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                #####
                out = self.model(x)
                loss = self.criterion(out, target)
                #####
                loss.backward()
                if cnt == 0:
                    for name, params in self.model.named_parameters():
                        params.grad.data.add_(params.data - self.global_model_params[name], alpha=self.alpha / self.epochs)
                        params.grad.data.add_(self.last_model[name].to(self.device) - params.data,
                                              alpha=self.alpha / self.gap / self.epochs)
                        # if len(list(params.size())) > 1:
                        #     params.grad.data.add_(-params.grad.data.mean(dim=tuple(range(1, len(list(params.grad.data.size())))), keepdim=True))
                else:
                    for (name, params), value in zip(self.model.named_parameters(), self.update_direction.values()):
                        params.grad.data.add_(params.data - self.global_model_params[name].to(self.device),
                                              alpha=self.alpha / self.epochs)
                        params.grad.data.add_(self.last_model[name].to(self.device) - params.data,
                                              alpha=self.alpha / self.gap / self.epochs)
                self.optimizer.step()
                batch_loss.append(loss.item())
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
        self.model = self.model_type(self.num_classes)
        self.prev_models = {x: self.model.cpu().state_dict() for x in range(self.args.client_number)}
        self.last_round = {x: 0 for x in range(self.args.client_number)}
        wandb.watch(self.model)

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.log_info(received_info, acc)
        if acc > self.acc:
            torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        for x in received_info:
            self.prev_models[x['client_index']] = x['weights']
            self.last_round[x['client_index']] = self.round
        server_outputs = [{'global': g, 'prev': self.prev_models, 'rounds': self.last_round} for g in server_outputs]
        return server_outputs

    def start(self):
        return [{'global': self.model.cpu().state_dict(), 'prev': self.prev_models, 'rounds': self.last_round} for x in
                range(self.args.thread_number)]
