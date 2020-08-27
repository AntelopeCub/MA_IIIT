import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import math
import random
import string

import h5py
import numpy as np
import tensorflow as tf

import data_loader
#import h5_util
import tfrecord
#from augment import add_augment, get_policies
from autoaugment import add_autoaugment, get_auto_policies


class Image_Generator(tf.keras.utils.Sequence):

    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        aug_pol,
        x_mean=None,
        x_std=None,
        pre_mode='norm'
        ):

        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.policies = get_auto_policies(aug_pol)
        self.aug_pol = aug_pol
        self.pre_mode = pre_mode
        #self._set_probs()

        if x_mean is None:
            self.x_mean = np.mean(x_set).astype('float32')
        else:
            self.x_mean = x_mean

        if x_std is None:
            self.x_std = np.std(x_set).astype('float32')
        else:
            self.x_std = x_std

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, step_idx):
        
        y_batch = self.y_set[self.batch_size * step_idx : self.batch_size * (step_idx + 1)]

        x_batch = []
        for x_idx, idx in enumerate(range(self.batch_size * step_idx, min(self.batch_size * (step_idx + 1), len(self.x_set)))):
            x = np.copy(self.x_set[idx])
            new_policy = creat_new_policy(self.policies, self.aug_pol)
            x = add_autoaugment(x, new_policy)
            x = np.asarray(x, dtype=np.float32)
            x_batch.append(x)

        #x_batch = np.asarray(x_batch)
        #x_batch = (x_batch - self.x_mean) / (self.x_std + 1e-7)
        x_batch = preprocess_input(x_batch, self.x_mean, self.x_std, mode=self.pre_mode)
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

def preprocess_input(img, img_mean, img_std, mode='norm'):
    if mode == 'norm':
        img = (np.array(img, dtype=np.float32) - img_mean) / (img_std + 1e-7)
    elif mode == 'scale':
        img = (np.array(img, dtype=np.float32) - 128.) / 32.
    else:
        raise Exception('Unknown preprocess mode: %s' % mode)
    return img
        
def set_temp_dataset(dataset, load_mode, aug_pol, pre_mode='norm'):
    x_train, y_train, _, _ = data_loader.load_data(dataset, load_mode=load_mode)
    x_mean = np.mean(x_train).astype('float32')
    x_std = np.std(x_train).astype('float32')
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    y_train = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train

    temp_file_name = 'temp_' + dataset + '_' + aug_pol + '_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=3)) + '.tfrecord'
    temp_file_path = 'd:/dataset/temp/' + temp_file_name

    policies = get_auto_policies(aug_pol)    
    x_train_aug = []
    for idx in range(len(x_train)):
        x = np.copy(x_train[idx])
        new_policy = creat_new_policy(policies, aug_pol)
        x = add_autoaugment(x, new_policy)
        #x = np.asarray(x, dtype=np.float32)
        #x = (x - x_mean) / (x_std + 1e-7)
        x = preprocess_input(x, x_mean, x_std, mode=pre_mode)
        x_train_aug.append(x)

    x_train_aug = np.asarray(x_train_aug)

    '''
    f = h5py.File(temp_file_path, 'w')
    h5_util.write_list(f, 'x_train', x_train_aug)
    h5_util.write_list(f, 'y_train', y_train)
    f.close()
    '''

    tfrecord.write_record(temp_file_path, x_train_aug, y_train)

    return temp_file_path

def load_temp_dataset(temp_file_path):
    '''
    f = h5py.File(temp_file_path, 'r')

    x_train = h5_util.read_list(f, 'x_train')
    x_train = [np.asarray(x_data, dtype='float32') for x_data in x_train]
    y_train = h5_util.read_list(f, 'y_train')
    y_train = [np.asarray(y_data, dtype='float32') for y_data in y_train]

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    f.close()
    #os.remove(temp_file_path)
    '''

    temp_file_list = glob.glob(temp_file_path + '*')
    assert len(temp_file_list) > 0, 'Temp dataset is missing, please check!'
    assert int(temp_file_list[0].split('-')[-1]) == len(temp_file_list), 'Several temp records are missing, please check!'
    x_train, y_train = tfrecord.extract_record(temp_file_list)
    y_train = tf.keras.utils.to_categorical(y_train)

    return x_train, y_train

def remove_temp_dataset(temp_file_path):
    temp_file_list = glob.glob(temp_file_path + '*')
    for temp_file in temp_file_list:
        os.remove(temp_file)

def creat_new_policy(policies, aug_pol):

    if 'base' in aug_pol:
        new_policy = []
    else:
        new_policy = policies[np.random.randint(len(policies))]
        
    if 'cifar' in aug_pol:
        new_policy = [{'op': 'mrx', 'prob': 0.5, 'mag': 0}, {'op': 'p&c', 'prob': 1.0, 'mag': 0}] + list(new_policy)
        new_policy = new_policy + [{'op': 'cut', 'prob': 1, 'mag': 5}]

    elif 'svhn' in aug_pol:
        new_policy = list(new_policy) + [{'op': 'cut', 'prob': 1.0, 'mag': 6.25}]

    else:
        raise Exception('Unknown policy: %s' % (aug_pol))

    return new_policy


if __name__ == "__main__":
    
    #path = set_temp_dataset('svhn_equal', 'tfrd', 'svhn_auto')
    #print(path)
    #x_train, y_train = load_temp_dataset('d:/dataset/temp/temp_svhn_equal_svhn_auto_jw0.tfrecord')
    remove_temp_dataset('d:/dataset/temp/temp_svhn_equal_svhn_auto_jw0.tfrecord')
