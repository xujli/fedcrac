import torch
import wandb
from methods.base import Base_Client, Base_Server

import random
from collections import defaultdict

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,# nesterov=True,
                                         weight_decay=self.args.wd)

    def client_selection(self):
        mapping_dict = defaultdict(list)
        if self.args.client_sample < 1.0:
            num_clients = int(self.args.client_number * self.args.client_sample)
            client_list = random.sample(range(self.args.client_number), num_clients)
        else:
            num_clients = self.args.client_number
            client_list = list(range(num_clients))

        if num_clients % self.args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / self.args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)

        return mapping_dict

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
