import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tdfs


def load_data(batch_size=128):
    x_train_list = []
    y_train_list = []
    train_data, test_data = tdfs.load(name='cifar10', split=['train', 'test'], batch_size=batch_size, shuffle_files=False)

    if batch_size != -1:
        for data in tdfs.as_numpy(train_data):
            data_tmp = data['image'].astype('float32') / 255
            data_tmp_mean = np.mean(data_tmp, axis=0)
            data_tmp -= data_tmp_mean
            x_train_list.append(data_tmp)
            y_train_list.append(tf.keras.utils.to_categorical(data['label'], 10))
    else:
        data = tdfs.as_numpy(train_data)
        data_tmp = data['image'].astype('float32') / 255
        data_tmp_mean = np.mean(data_tmp, axis=0)
        data_tmp -= data_tmp_mean
        x_train_list = data_tmp
        y_train_list = tf.keras.utils.to_categorical(data['label'], 10)

    return x_train_list, y_train_list

if __name__ == "__main__":

    x_train_list, y_train_list = load_data(batch_size=128)

    a = 1