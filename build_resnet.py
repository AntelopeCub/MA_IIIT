import tensorflow as tf
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense,
                                     Flatten, GlobalAveragePooling2D, Input)
from tensorflow.keras.models import Model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 l2_reg=None):
    """
    2D Convolution-Batch Normalization-Activation stack builder
    See: https://keras.io/examples/cifar10_resnet/
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2_reg)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, l2_reg=None):
    """
    ResNet Version 1 Model builder [a]
    See: https://keras.io/examples/cifar10_resnet/
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, l2_reg=l2_reg)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             l2_reg=l2_reg)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=True,
                                 l2_reg=l2_reg)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    #x = AveragePooling2D(pool_size=8)(x)
    #y = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_reg)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10, l2_reg=None):
    """
    ResNet Version 2 Model builder [b]
    See: https://keras.io/examples/cifar10_resnet/
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                     l2_reg=l2_reg)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             l2_reg=l2_reg)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=True,
                                 l2_reg=l2_reg)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = AveragePooling2D(pool_size=8)(x)
    #y = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_reg)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def ResNet56(input_shape=(32,32,3), num_class=10, l2_reg=None):
    return resnet_v1(input_shape=input_shape, depth=56, num_classes=num_class, l2_reg=l2_reg)

if __name__ == "__main__":
    model = ResNet56(l2_reg=tf.keras.regularizers.l2(l=5e-4))
    a = 1
