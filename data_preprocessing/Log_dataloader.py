import os
import logging
import numpy as np
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from data_preprocessing.Log_dataset import Log_truncated

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch
    
def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for seq, label in batch:
        tensors += [seq]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts



def load_data(datadir):
    if datadir.split('/')[-1] in ['BGL', 'Spirit', 'ThunderBird']:
        train_ds = Log_truncated(datadir, train=True, prefix='bert')
        test_ds = Log_truncated(datadir, train=False, prefix='bert')
    elif datadir.split('/')[-1] in ['HDFS']:
        train_ds = Log_truncated(datadir, train=True, prefix='neural')
        test_ds = Log_truncated(datadir, train=False, prefix='neural')

    y_train, y_test = train_ds.target, test_ds.target
    del train_ds
    del test_ds
    return (y_train, y_test)

def partition_data(datadir, partition, n_nets, alpha, silos=5):
    logging.info("*********partition data***************")
    y_train, y_test = load_data(datadir)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    class_num = 2

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == 'orthogonal':
        net_dataidx_map = {}
        data_num = n_train // silos
        classes_silos = np.arange(0, class_num)
        for i in range(silos):
            if i != silos - 1:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: len(classes_silos) // silos * (1 + i)]
            else:
                classes_ids = \
                    classes_silos[len(classes_silos) // silos * i: -1]
            idx_i = []
            for class_id in classes_ids:
                idx_k = np.where(y_train == class_id)[0]
                idx_i.extend(idx_k)
            np.random.shuffle(idx_i)
            batch_idxs = np.array_split(idx_i, n_nets // silos)
            for j in range(n_nets // silos):
                net_dataidx_map[i * n_nets // silos + j] = batch_idxs[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):

    if datadir.split('/')[-1] in ['BGL', 'Spirit', 'ThunderBird']:
        train_ds = Log_truncated(datadir, train=True, prefix='bert', dataidxs=dataidxs)
        test_ds = Log_truncated(datadir, train=False, prefix='bert')
    elif datadir.split('/')[-1] in ['HDFS']:
        train_ds = Log_truncated(datadir, train=True, prefix='neural', dataidxs=dataidxs)
        test_ds = Log_truncated(datadir, train=False, prefix='neural')
        
    workers = 0
    persist = False

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=workers, #collate_fn=collate_fn,
                               )
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=workers,# collate_fn=collate_fn,
                              )

    return train_dl, test_dl


def load_partition_data(data_dir, partition_method, partition_alpha, client_number, batch_size, silos):
    # if not os.path.exists(
    #         'processed_{}_{}_{}_{}_{}'.format(data_dir.split('/')[1], partition_method, partition_alpha, client_number,
    #                                           batch_size)):
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(data_dir, partition_method, client_number,
                                                                      partition_alpha, silos)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(data_dir, 5, 5)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(data_dir, batch_size, batch_size, dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
           
           
if __name__ == '__main__':
    train_data_num, test_data_num, train_data_global, test_data_global, \
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = load_partition_data('../data/Log/embeddings/BGL', 'hetero', 0.5, 10, 32, 5)

    for item in train_data_global:
        print(item[0].shape)
