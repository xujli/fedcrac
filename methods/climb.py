import torch
import wandb
import numpy as np
from methods.base import Base_Client, Base_Server


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9 if args.momentum else 0,#, nesterov=True,
                                         weight_decay=self.args.wd)
       
class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        wandb.watch(self.model)
        self.tolerance_epsilon = 1.
        self.lambda_var = torch.zeros(self.args.client_number).to(self.device)
        self.lambda_lr = 1.
        self.global_lr = 1.

    def client_test(self, dl):
        self.model.to(self.device)
        self.model.eval()
        hs = None
        labelss = None
        preds = None
        logits = None

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dl):
                x = x.to(self.device)
                target = target.to(self.device).long()

                pred = self.model(x)
                if type(pred) == tuple:
                    h, pred = pred
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                hs = h.detach() if hs is None else torch.cat([hs, h.detach().clone()], dim=0)
                labelss = target if labelss is None else torch.cat([labelss, target.clone()], dim=0)
                logits = pred.detach() if logits is None else torch.cat([logits, pred.detach().clone()], dim=0)
                preds = predicted.detach() if preds is None else torch.cat([preds, predicted.detach().clone()], dim=0)

            loss = (test_loss / test_sample_number)
        return loss

    def operations(self, client_info):

        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        client_idx = [c['client_index'] for c in client_info]
        # calculates weights
        weights = (1. + self.lambda_var - torch.mean(self.lambda_var))
        client_losses = torch.tensor([self.client_test(dl) for i, dl in self.train_data.items()]).to(self.device)
        lambda_new = self.lambda_var + self.lambda_lr * (client_losses - torch.mean(client_losses) - self.tolerance_epsilon) / self.args.client_number
        self.lambda_var = torch.clamp(lambda_new, min=0., max=100.)
        
        ssd = self.model.cpu().state_dict()
        for key in ssd:
            ssd[key] += sum([(sd[key] - ssd[key]) * weights[idx].cpu() * self.global_lr / len(client_idx) for idx, sd in zip(client_idx, client_sd)])
        self.model.load_state_dict(ssd)
        
        client_acc = np.mean([c['acc'] for c in client_info])
        self.round += 1
        self.get_mean_var(client_info)
        return [self.model.cpu().state_dict() for _ in range(self.args.thread_number)], client_acc