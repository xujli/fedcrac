import os
import torch
import pickle
import logging
import numpy as np
import torch.utils.data as data

# import tensorflow as tf

# # 设置GPU显存使用量
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class Log_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=False, prefix='neural'):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.prefix = prefix
        
        data = np.load('{}/{}-{}-data.npy'.format(self.root, self.prefix, 'train' if self.train else 'train'), allow_pickle=True)
        label = np.load('{}/{}-{}-label.npy'.format(self.root, self.prefix, 'train' if self.train else 'train'), allow_pickle=True)
        print(data.shape, self.train)
        if self.dataidxs is not None:
            self.data = data[self.dataidxs]
            self.target = label[self.dataidxs]
        else:
            self.data = data
            self.target = label

    # def __build_truncated_dataset__(self):
        # with open('{}/{}-{}.pkl'.format(self.root, self.prefix, 'train' if self.train else 'test'), 'rb') as f:
        #     data, label = pickle.load(f)
        #     new_data = []
        #     for i, item in enumerate(data):
        #         tmp = []
        #         for arr in item:
        #             arr = arr.numpy().astype(np.float32)
        #             tmp.append(arr)
        #         new_data.append(tmp)
        #         if (i == 99999) and self.train:
        #             break
        #     data = new_data[:100000]
        #     label = label[:100000]
        # #     new_label = label
            
        #     np.save('{}/{}-{}-data.npy'.format(self.root, self.prefix, 'train' if self.train else 'test'), np.array(data))
        #     np.save('{}/{}-{}-label.npy'.format(self.root, self.prefix, 'train' if self.train else 'test'), np.array(label))

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = np.array(self.data[index]), self.target[index]
        
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
        dataset = Log_truncated('../data/Log/embeddings/HDFS')
        for item in dataset:
            print(item[0].shape)