import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image

'''
gpus = tf.config.experimental.list_physical_devices('GPU') #limit gpu memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''

def VGG16_BN(input_shape, l2_reg):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    return model

def VGG9_no_BN(input_shape, l2_reg):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg, input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2_reg))
    #model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    return model

def VGG9_BN(input_shape, l2_reg):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    return model

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
        model_vgg9_bn = VGG9_BN(input_shape, l2_reg)
        model_save_path = './models/vgg9/vgg9_bn.h5'
        model_vgg9_bn.compile(optimizer=SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        print(model_vgg9_bn.summary())

        history = model_vgg9_bn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
        model_vgg9_bn.save(model_save_path)

    elif model_type == 'vgg9':
        model_vgg9 = VGG9_no_BN(input_shape, None)
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