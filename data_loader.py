import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image

dataset_root_path = 'd:/dataset/'

def load_data(dataset, load_mode='tfds'):

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    
    if load_mode == 'tfds':
        train_data, test_data = tfds.load(name=dataset, split=['train', 'test'], batch_size=-1, shuffle_files=True)
        train_data = tfds.as_numpy(train_data)
        test_data = tfds.as_numpy(test_data)
        x_train_list = train_data['image'].astype('float32')
        x_test_list = test_data['image'].astype('float32')
        y_train_list = tf.keras.utils.to_categorical(train_data['label'])
        y_test_list = tf.keras.utils.to_categorical(test_data['label'])

    elif load_mode == 'path':
        dataset_path = dataset_root_path + dataset
        assert os.path.exists(dataset_path), 'Dataset may not exist, please check the path: ' + dataset_path
        train_path = dataset_path + '/train/'
        test_path = dataset_path + '/test/'
        classes = os.listdir(train_path)

        gc.disable()
        for idx, cl in enumerate(classes):
            data_imgs = os.listdir(train_path + cl)
            img_load = [image.load_img(train_path + cl + '/' + data_img) for data_img in data_imgs]
            x_train_list.extend([image.img_to_array(img) for img in img_load])
            y_train_list.extend([idx for i in range(len(data_imgs))])
            
            data_imgs = os.listdir(test_path + cl)
            img_load = [image.load_img(test_path + cl + '/' + data_img) for data_img in data_imgs]
            x_test_list.extend([image.img_to_array(img) for img in img_load])
            y_test_list.extend([idx for i in range(len(data_imgs))])
        gc.enable()

        x_train_list = np.array(x_train_list, dtype=np.float32)
        x_test_list = np.array(x_test_list, dtype=np.float32)
        y_train_list = tf.keras.utils.to_categorical(y_train_list)
        y_test_list = tf.keras.utils.to_categorical(y_test_list)
        x_train_list, y_train_list, x_test_list, y_test_list = shuffle_data(x_train_list, y_train_list, x_test_list, y_test_list)
        
    return x_train_list, y_train_list, x_test_list, y_test_list
            
def shuffle_data(x_train_list, y_train_list, x_test_list, y_test_list):
    train_shuffle = np.arange(x_train_list.shape[0])
    test_shuffle = np.arange(x_test_list.shape[0])
    np.random.shuffle(train_shuffle)
    np.random.shuffle(test_shuffle)
    x_train_list = x_train_list[train_shuffle]
    y_train_list = y_train_list[train_shuffle]
    x_test_list = x_test_list[test_shuffle]
    y_test_list = y_test_list[test_shuffle]
    return x_train_list, y_train_list, x_test_list, y_test_list


'''
def load_data(batch_size=128):
    x_train_list = []
    y_train_list = []
    train_data, test_data = tfds.load(name='cifar10', split=['train', 'test'], batch_size=batch_size, shuffle_files=False)

    if batch_size != -1:
        for data in tfds.as_numpy(train_data):
            data_tmp = data['image'].astype('float32') / 255
            data_tmp_mean = np.mean(data_tmp, axis=0)
            data_tmp -= data_tmp_mean
            x_train_list.append(data_tmp)
            y_train_list.append(tf.keras.utils.to_categorical(data['label'], 10))
    else:
        data = tfds.as_numpy(train_data)
        data_tmp = data['image'].astype('float32') / 255
        data_tmp_mean = np.mean(data_tmp, axis=0)
        data_tmp -= data_tmp_mean
        x_train_list = data_tmp
        y_train_list = tf.keras.utils.to_categorical(data['label'], 10)

    return x_train_list, y_train_list
'''

if __name__ == "__main__":

    import time

    start_time = time.time()

    x_train_list, y_train_list, x_test_list, y_test_list = load_data('cifar10', load_mode='tfds')
    mid_time = time.time()
    print(mid_time - start_time)

    x_train_list, y_train_list, x_test_list, y_test_list = load_data('cifar10', load_mode='path')
    end_time = time.time()
    print(end_time - mid_time)

    a = 1