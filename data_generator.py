import math
import operator
import os
import random
import string
import threading

import h5py
import numpy as np
import tensorflow as tf

import data_loader
import h5_util
from augment import add_augment, get_policies


class Image_Generator(tf.keras.utils.Sequence):

    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        policy_list
        ):

        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.policies = [get_policies(policy) for policy in policy_list]
        self._set_probs()

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, step_idx):
        
        y_batch = self.y_set[self.batch_size * step_idx : self.batch_size * (step_idx + 1)]

        new_policies = []
        for p_idx in range(len(self.policies)):
            choose = np.random.choice(range(len(self.policies[p_idx])), len(y_batch), p=self.probs[p_idx])
            new_policies.append(operator.itemgetter(*choose)(self.policies[p_idx]))

        x_batch = []
        for x_idx, idx in enumerate(range(self.batch_size * step_idx, self.batch_size * (step_idx + 1) if self.batch_size * (step_idx + 1) < len(self.x_set) else len(self.x_set))):
            x = np.copy(self.x_set[idx])
            new_policy = [p[x_idx] for p in new_policies]
            x = add_augment(x, new_policy)
            x = np.asarray(x, dtype=np.float32) / 255.0
            x_batch.append(x)

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        return x_batch, y_batch
        
    def on_epoch_end(self):
        self._shuffle()

    def _set_probs(self):
        self.probs = []
        for policy in self.policies:
            self.probs.append([p.get('prob') for p in policy])

    def _shuffle(self):
        shuffle_list = np.arange(self.x_set.shape[0])
        np.random.shuffle(shuffle_list)
        self.x_set = self.x_set[shuffle_list]
        self.y_set = self.y_set[shuffle_list]
        
def set_temp_dataset(dataset, load_mode, aug_pol):
    x_train, y_train, _, _ = data_loader.load_data(dataset, load_mode=load_mode)
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    if aug_pol == 'baseline':
        policy_list = ['reduced_mirror',  'crop', 'cutout']
    elif aug_pol == 'cifar_pol':
        policy_list = ['cifar_pol1', 'cifar_pol2']

    policies = [get_policies(policy) for policy in policy_list]
    probs = []
    for policy in policies:
        probs.append([p.get('prob') for p in policy])
    
    new_policies = []
    for p_idx in range(len(policies)):
        choose = np.random.choice(range(len(policies[p_idx])), len(x_train), p=probs[p_idx])
        new_policies.append(operator.itemgetter(*choose)(policies[p_idx]))
    
    x_train_aug = []
    for idx in range(len(x_train)):
        x = np.copy(x_train[idx])
        new_policy = [p[idx] for p in new_policies]
        x = add_augment(x, new_policy)
        x = np.asarray(x, dtype=np.float32) / 255.0
        x_train_aug.append(x)

    temp_file_name = 'temp_' + dataset + '_' + aug_pol + '_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3)) + '.h5'
    temp_file_path = './models/' + temp_file_name

    f = h5py.File(temp_file_path, 'w')
    h5_util.write_list(f, 'x_train', x_train_aug)
    h5_util.write_list(f, 'y_train', y_train)

    f.close()

    return temp_file_path

def load_temp_dataset(temp_file_path):
    f = h5py.File(temp_file_path, 'r')

    x_train = h5_util.read_list(f, 'x_train')
    x_train = [np.asarray(x_data, dtype='float32') for x_data in x_train]
    y_train = h5_util.read_list(f, 'y_train')
    y_train = [np.asarray(y_data, dtype='float32') for y_data in y_train]

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    f.close()
    #os.remove(temp_file_path)

    return x_train, y_train