import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, InputLayer,
                                     MaxPooling2D, Softmax)
from tensorflow.keras.models import Sequential

from quantization.Q_ActivationNormLayers import ActivationLayer_signed
from quantization.Q_Conv2dNorm import Conv2dNorm
from quantization.Q_RegularizersNorm import max_weights_reg


def VGG16_BN(input_shape, l2_reg, num_class=10, fc_type=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    if fc_type is None:
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    elif fc_type == 'dropout':
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
        model.add(Dropout(0.5))
    elif fc_type == 'avg':
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
   
    model.add(Dense(num_class, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    
    return model

def VGG9_no_BN(input_shape, l2_reg, num_class=10, fc_type=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2_reg))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2_reg))
    model.add(Dense(num_class, activation='softmax'))
    
    return model

def VGG9_BN(input_shape, l2_reg, num_class=10, fc_type=None):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    if fc_type is None:
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    elif fc_type == 'dropout':
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
        model.add(Dropout(0.5))
    elif fc_type == 'avg':
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.5))
    
    model.add(Dense(num_class, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2_reg))

    return model

def VGG9_QN(input_shape, l2_reg_rate=0, num_class=10, L_A=[3, 5], L_W=[1, 7]):
    if l2_reg_rate != 0:
        l2_reg = regularizers.l2(l=l2_reg_rate)
    else:
        l2_reg = None
    model = Sequential()
    model.add(Conv2dNorm(64, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W,
                         input_shape=input_shape))
    model.add(Conv2dNorm(64, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(128, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(128, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(num_class, name='c1x1_2d',
                     kernel_size=(1,1),
                     use_bias=False,
                     kernel_initializer='he_normal',
                     kernel_regularizer=max_weights_reg(L_W=L_W, l2=l2_reg_rate)))

    model.add(ActivationLayer_signed(L_A=[L_A[0], L_A[1]]))

    model.add(GlobalAveragePooling2D())
    #model.add(Softmax())

    return model

def VGG16_QN(input_shape, l2_reg_rate=0, num_class=10, L_A=[3, 5], L_W=[1, 7]):
    if l2_reg_rate != 0:
        l2_reg = regularizers.l2(l=l2_reg_rate)
    else:
        l2_reg = None
    model = Sequential()
    model.add(Conv2dNorm(64, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W,
                         input_shape=input_shape))
    model.add(Conv2dNorm(64, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(128, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(128, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(256, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(Conv2dNorm(512, kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2_reg,
                         L_A=L_A,
                         L_W=L_W))
    model.add(MaxPooling2D())

    model.add(ActivationLayer_signed(L_A=[L_A[0], L_A[1]]))

    model.add(GlobalAveragePooling2D())
    #model.add(Softmax())

    return model



if __name__ == "__main__":
    #model = VGG16_BN((32,32,3), None, fc_type='avg')
    from tensorflow.keras import regularizers
    model = VGG16_QN((32,32,3), l2_reg_rate=5e-4, num_class=10, L_A=[3, 5], L_W=[1, 7])
    model.summary()
