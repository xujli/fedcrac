import torch
import wandb
from methods.base import Base_Client, Base_Server


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,
                                         weight_decay=self.args.wd)


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
        self.mu = args.mu # Threshold


    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
        ssd = self.model.state_dict()
        flatten_sd = self._flatten_weights_from_sd(self.model, self.device)
        flags = torch.zeros(len(cw))
        for i, sd in enumerate(client_sd):
            flatten_csd = self._flatten_weights_from_sd(sd, self.device)
            similarity = (torch.sum((flatten_sd > 0) * (flatten_csd > 0)) + \
                          torch.sum((flatten_sd < 0) * (flatten_csd < 0))) / torch.numel(flatten_sd)
            if similarity > self.mu:
                flags[i] = 1


        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        self.round += 1
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)]