import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.layers import InputLayer, Conv2D, Dense, BatchNormalization, Flatten, MaxPooling2D
#from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler

import data_loader
import data_generator
from build_vgg import VGG9_BN, VGG9_no_BN
from build_resnet import ResNet56



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

class build_model(object):

    def __init__(self, model_type, dataset, l2_reg_rate=None):

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

        if model_type == 'vgg9':
            self.model = VGG9_no_BN(self.input_shape, self.l2_reg, num_class=self.num_class)
        elif model_type == 'vgg9_bn':
            self.model = VGG9_BN(self.input_shape, self.l2_reg, num_class=self.num_class)

    def train_model(self, learning_rate=0.1, batch_size=128, epochs=20, load_mode='tfds', plot_history=False, add_aug=False, aug_pol='baseline', callbacks=None):
        
        x_train, y_train, x_test, y_test = data_loader.load_data(self.dataset, load_mode=load_mode)
        if add_aug:
            if aug_pol == 'baseline':
                policy_list = ['reduced_mirror',  'crop', 'cutout']
            train_gen = data_generator.Image_Generator(x_train, y_train, batch_size, policy_list)

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        if 'vgg' in self.model_type:
            self.optimizer = SGD(learning_rate=learning_rate)
            self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])            
            '''
            logs = ".\logs\log" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                            histogram_freq = 1,
                                                            profile_batch = 1)
            '''
            if not add_aug:
                self.history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
            else:
                self.history = self.model.fit(train_gen, validation_data=(x_test, y_test), epochs=epochs, steps_per_epoch=len(train_gen), callbacks=callbacks)
            if plot_history:
                plot_result(self.history)

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

if __name__ == "__main__":
    
    # init value
    img_size = 32
    batch_size = 2048
    epochs = 50
    lr = 0.1
    subtract_pixel_mean = True
    model_type = 'vgg9'

    #load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    
    l2_reg = regularizers.l2(l=0.0005)

    if model_type == 'vgg9_bn':
        model_vgg9_bn = VGG9_BN(input_shape, None)
        model_save_path = './models/vgg9/vgg9_bn.h5'
        model_vgg9_bn.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        print(model_vgg9_bn.summary())

        history = model_vgg9_bn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
        model_vgg9_bn.save(model_save_path)

    elif model_type == 'vgg9':
        model_vgg9 = VGG9_no_BN(input_shape, l2_reg)
        model_save_path = './models/vgg9/vgg9.h5'
        model_vgg9.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        print(model_vgg9.summary())

        history = model_vgg9.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
        model_vgg9.save(model_save_path)



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
    #plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    ax1 = plt.subplot(211, sharex = ax2)
    plt.plot(epochs_idx[::1], loss[::1], 'bo', label='Training loss')
    plt.plot(epochs_idx[::1], val_loss[::1], 'b', label='Validation loss')
    plt.title('Training and validation loss / acc')
    #plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig1.savefig('./models/vgg9/vgg9_result.pdf', dpi=300, format='pdf')