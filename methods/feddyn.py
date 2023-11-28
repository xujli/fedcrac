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
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)
        self.mu = args.mu
        self.local_delta = {}

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        fixed_model = deepcopy(self.model)
        for param_t in fixed_model.parameters():
            param_t.requires_grad = False
        fixed_params = {n: p for n, p in fixed_model.named_parameters()}

        if not bool(self.local_delta):
            for n, p in fixed_params.items():
                self.local_delta[n] = torch.zeros(p.shape)

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                #####
                out = self.model(x)
                loss = self.criterion(out, target)

                ## Weight L2 loss
                reg_loss = 0
                for n, p in self.model.named_parameters():
                    reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                ## local gradient regularization
                lg_loss = 0
                for n, p in self.model.named_parameters():
                    p = torch.flatten(p)
                    local_d = self.local_delta[n].detach().clone().to(self.device)
                    local_grad = torch.flatten(local_d)
                    lg_loss += (p * local_grad.detach()).sum()

                #####
                loss = loss - lg_loss + 0.5 * self.mu * reg_loss
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        weights = self.model.cpu().state_dict()

        # Update Local Delta
        for n, p in self.model.cpu().named_parameters():
            self.local_delta[n] = (
                    self.local_delta[n] - self.mu * (p.data.cpu() - fixed_params[n].cpu()).detach().clone())
        return weights


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
        self.mu = args.mu
        self.coef = args.client_sample
        self.h = None

        wandb.watch(self.model)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]

        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.cpu().state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        if self.h is None:
            self.h = {}
            for (name, new), (_, old) in zip(ssd.items(), self.model.cpu().state_dict().items()):
                self.h[name] = - self.mu * self.coef * (new - old)
        else:
            for (name, new), (_, old) in zip(ssd.items(), self.model.cpu().state_dict().items()):
                self.h[name] = self.h[name] - self.mu * self.coef * (new - old)

        for name, delta in self.h.items():
            ssd[name] += - 1 / self.mu * delta

        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]
