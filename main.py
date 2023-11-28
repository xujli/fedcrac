'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import torch
import wandb
import numpy as np
import random
import data_preprocessing.data_loader as dl
import data_preprocessing.Log_dataloader as Log_dl
import argparse
import importlib
from config import add_args
from models.mobilenet import mobilenet
from models.resnet import resnet50, resnet18, resnet10, resnet8
from models.resnet_gradaug import resnet56 as resnet56_gradaug
from models.resnet_gradaug import resnet18 as resnet18_gradaug
from models.resnet_stochdepth import resnet56 as resnet56_stochdepth
from models.resnet_stochdepth import resnet18 as resnet18_stochdepth
from models.resnet_fedalign import resnet56 as resnet56_fedalign
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from models.net import SimpleCNN, modVGG, OneDCNN
from models.MLP import MLP
from models.lenet5 import Lenet5
from models.neurallog import NeuralLog, Head
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time
import sys

import data_preprocessing.custom_multiprocess as cm

np.set_printoptions(threshold=np.inf)
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["WANDB_API_KEY"] = 'c2837c6d9f0b2b836c42d0812fb9c4366d712d90'
# os.environ["WANDB_MODE"] = "offline"


# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])


def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number * args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_random_seed(args.seed)
    # wandb initialization
    experiment_name = 'seed:' + str(args.seed)
    print(experiment_name)
    wandb_log_dir = os.path.join('./fed/wandb', experiment_name)
    if not os.path.exists('{}'.format(wandb_log_dir)):
        os.makedirs('{}'.format(wandb_log_dir))
    odds = 'LT' if args.LT else ""
    wandb.init(entity='lxjxlxj', project='FLMC_exp' + ('_M' if args.momentum else ''),
               group=args.data_dir.split('/')[-1] + odds + "_" + args.partition_method + "_" + args.net + "_" + (
                   str(args.partition_alpha) if args.partition_method == 'hetero' else "") \
                     + '_epochs:' + str(args.epochs) + '_client:' + str(args.client_number) + '_fraction:' \
                     + str(args.client_sample),
               job_type=args.method + (
                   "_" + args.additional_experiment_name if args.additional_experiment_name != '' else ''))

    wandb.run.name = experiment_name + "_" + str(args.mu) + ('_imb:{:.1f}'.format(args.imbalance_ratio) if args.LT else '')
    wandb.run.save()
    wandb.config.update(args)

    # get data
    if args.LT:
        _, _, train_data_global, test_data_global, _, train_data_local_dict, test_data_local_dict, class_num = \
            dl.load_partition_data_LT(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.silos_number, imb_ratio=args.imbalance_ratio)
    elif args.Log:
        _, _, train_data_global, test_data_global, _, train_data_local_dict, test_data_local_dict, class_num = \
            Log_dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.silos_number)
    else:
        _, _, train_data_global, test_data_global, _, train_data_local_dict, test_data_local_dict, class_num = \
            dl.load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size, args.silos_number)

    mapping_dict = allocate_clients_to_threads(args)
    def import_class(module_name, class_name):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    # init method and model type
    Server = import_class('methods.{}'.format(args.method), 'Server')
    Client = import_class('methods.{}'.format(args.method), 'Client')

    if args.method in ['fedavg', 'fedavgweighted', 'fedratio', 'focal', 'fedmargin', 'fedgmmargin', 'fedgmmargin2', 'fedgmmargin3', 
                       'fedgmmargin4', 'fedgmmargin5', 'fedgmmargin6', 'fedcos', 'dmfl', 'dmfl2', 'dmfl3', 'fedlc', 'ccvr', 'fedrev', 
                       'creff', 'fedprox', 'moon', 'feddebias', 'feddyn', 'fedrs', 'mixup', 'naivemix', 'globalmix', 'fedmix', 'balancefl']:
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
        
    elif args.method == 'climb':
        Model = eval(args.net)
        server_dict = {'train_data': train_data_local_dict, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'gradaug':
        Model = eval(args.net + '_gradaug')
        width_range = [args.width, 1.0]
        resolutions = [32, 28, 24, 20] if 'cifar' in args.data_dir else [224, 192, 160, 128]
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    elif args.method == 'fedtrip':
        Model = eval(args.net)
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(), 'epochs': args.epochs,
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'stochdepth':
        Model = eval(args.net + '_stochdepth')
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in
                       range(args.thread_number)]
    elif args.method == 'fedalign':
        Model = eval(args.net + '_fedalign')
        width_range = [args.width, 1.0]
        resolutions = [32] if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict,
                        'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')

    # init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args, i+1))

    # Start server and get initial outputs
    pool = cm.DreamPool(args.thread_number, init_process, (client_info, Client))
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}'.format(os.getcwd(),
                                                               time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs,
                                                               args.client_number)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    server = Server(server_dict, args)
    server_outputs = server.start()
    # Start Federated Training
    time.sleep(40 * (args.client_number / 16))  # Allow time for threads to start up
    for r in range(args.comm_round):
        logging.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = pool.map(run_clients, server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        server_outputs = server.run(client_outputs)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end - round_start))

    server.finalize()
    pool.close()
    pool.join()
    wandb.finish()
