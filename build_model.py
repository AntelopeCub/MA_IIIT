import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD

import data_generator
import data_loader
from build_resnet import ResNet56
from build_vgg import VGG9_BN, VGG9_no_BN, VGG16_BN


def lr_decay(epoch):
    init_lr = 0.01
    factor = 0.1
    if epoch < 150:
        return init_lr
    elif epoch < 225:
        return init_lr * (factor ** 1)
    elif epoch < 275:
        return init_lr * (factor ** 2)
    else:
        return init_lr * (factor ** 3)


def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_idx = range(1, len(acc) + 1)

    fig1 = plt.figure('Training Result', figsize = (8, 10))
    plt.figure('Training Result')
    ax2 = plt.subplot(212)
    plt.plot(epochs_idx[::1], acc[::1], 'bo', label='Training acc')
    plt.plot(epochs_idx[::1], val_acc[::1], 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    ax1 = plt.subplot(211, sharex = ax2)
    plt.plot(epochs_idx[::1], loss[::1], 'bo', label='Training loss')
    plt.plot(epochs_idx[::1], val_loss[::1], 'b', label='Validation loss')
    plt.title('Training and validation loss / acc')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


class Cyclic_LR_Scheduler():
    
    def __init__(self, initial_rate, cycle_length=33, end_rate=1e-10, drop_rate=1.):
        self.initial_rate = initial_rate
        self.cycle_length = cycle_length
        self.end_rate     = end_rate
        self.drop_rate    = drop_rate
        
    def cyc_decay(self, epoch):
        return self.end_rate + 0.5* self.initial_rate * ( 1+np.cos(np.mod(epoch, self.cycle_length+1)*np.pi/(self.cycle_length+1)) ) * self.drop_rate**(np.floor(epoch / self.cycle_length))


class build_model(object):

    def __init__(self, model_type, dataset, l2_reg_rate=None, fc_type=None):

        self.model_type = model_type

        if dataset in ['cifar10', 'svhn_equal']:
            self.num_class = 10
            self.input_shape = (32, 32, 3)
        elif dataset in ['cifar100']:
            self.num_class = 100
            self.input_shape = (32, 32, 3)

        self.dataset = dataset

        if l2_reg_rate is not None:
            self.l2_reg = regularizers.l2(l=l2_reg_rate)
        else:
            self.l2_reg = None

        self.fc_type = fc_type

        if model_type == 'vgg9':
            self.model = VGG9_no_BN(self.input_shape, self.l2_reg, num_class=self.num_class)
        elif model_type == 'vgg9_bn':
            self.model = VGG9_BN(self.input_shape, self.l2_reg, num_class=self.num_class)
        elif model_type == 'vgg16_bn':
            self.model = VGG16_BN(self.input_shape, self.l2_reg, num_class=self.num_class, fc_type=self.fc_type)

    def train_model(self, optimizer=None, batch_size=128, epochs=20, load_mode='tfds', plot_history=False, add_aug=False, aug_pol='baseline', callbacks=None, workers=1):
        
        if optimizer is None:
            self.optimizer = SGD(learning_rate=0.1)
        else:
            self.optimizer = optimizer

        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        x_train, y_train, x_test, y_test = data_loader.load_data(self.dataset, load_mode=load_mode)
        x_mean = np.mean(x_train).astype('float32')
        x_std = np.std(x_train).astype('float32')
        if add_aug:
            if aug_pol == 'baseline':
                policy_list = ['reduced_mirror',  'crop', 'cutout']
            elif aug_pol == 'cifar_pol':
                policy_list = ['cifar_pol1', 'cifar_pol2']
            train_gen = data_generator.Image_Generator(x_train, y_train, batch_size, policy_list, x_mean=x_mean, x_std=x_std)
        else:
            x_train = (x_train.astype('float32') - x_mean) / (x_std + 1e-7)
        
        x_test = (x_test.astype('float32') - x_mean) / (x_std + 1e-7)

        if 'vgg' in self.model_type:
            #self.optimizer = SGD(learning_rate=learning_rate)
            #self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            if not add_aug:
                self.history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks, workers=workers)
            else:
                self.history = self.model.fit(train_gen, validation_data=(x_test, y_test), epochs=epochs, steps_per_epoch=len(train_gen), callbacks=callbacks, workers=workers)
            if plot_history:
                plot_result(self.history)
        '''
        elif 'resnet' in self.model_type:
            self.optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            callback = [LearningRateScheduler(lr_decay)]
            if not add_aug:
                self.history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
            else:
                self.history = self.model.fit(train_gen, validation_data=(x_test, y_test), epochs=epochs, steps_per_epoch=len(train_gen), callbacks=callbacks)
            if plot_history:
                plot_result(self.history)
        '''
        
if __name__ == "__main__":
    
    pass
