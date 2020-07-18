from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, InputLayer,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential


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
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2_reg))
    model.add(Dense(num_class, activation='softmax'))
    return model


if __name__ == "__main__":
    
    model = VGG16_BN((32,32,3), None, fc_type='avg')
    model.summary()