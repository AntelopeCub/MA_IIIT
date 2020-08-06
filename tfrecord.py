import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import math
import time
import timeit
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import data_loader

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "shape": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "imgdt": tf.io.FixedLenFeature([], tf.string),
    "labdt": tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

def write_record(file_path, x_set, y_set):

    split_size = 65536
    split_num = math.ceil(len(x_set) / split_size)

    for i in range(split_num):
        file_path_split = file_path + '-%05d-of-%05d' % (i, split_num)        
        with tf.io.TFRecordWriter(file_path_split) as file_writer:
            for idx in range(split_size*i, min(split_size*(i+1), len(x_set))):
                record_bytes = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_set[idx].tobytes()])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[y_set[idx]])),
                    "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(x_set[idx].shape))),
                    "imgdt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(x_set[idx].dtype).encode()])),
                    "labdt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(y_set[idx].dtype).encode()])),
                })).SerializeToString()
                file_writer.write(record_bytes)

def read_record(file_path, dataset):
    assert os.path.exists(file_path), 'Dataset does not exsist, please check path: ' + file_path
    train_files = glob.glob(file_path + dataset + '-train.tfrecord*')
    test_files = glob.glob(file_path + dataset + '-test.tfrecord*')

    assert (len(train_files) > 0 and len(test_files) > 0), 'Train/Test data is missing, please check!'
    assert (int(train_files[0].split('-')[-1]) == len(train_files) and int(test_files[0].split('-')[-1]) == len(test_files)), 'Several Train/Test records are missing, please check!'

    train_image, train_label = extract_reccord(train_files)
    test_image, test_label = extract_reccord(test_files)

    return train_image, train_label, test_image, test_label

def extract_reccord(file_list):
    image_list = []
    label_list = []
    for record_file in file_list:
        raw_record = tf.data.TFRecordDataset(record_file)
        parsed_record = raw_record.map(_parse_function)
        for features in parsed_record:
            image = np.frombuffer(features['image'].numpy(), dtype=features['imgdt'].numpy().decode()).reshape(features['shape'].numpy())
            label = features['label'].numpy()
            image_list.append(image)
            label_list.append(label)
    
    return np.asarray(image_list), np.asarray(label_list)



if __name__ == "__main__":
    '''
    train_path = os.path.join('D:/dataset/temp/', "cifar10-train.tfrecord")
    test_path = os.path.join('D:/dataset/temp/', "cifar10-test.tfrecord")

    train_data, test_data = tfds.load(name='cifar10', split=['train', 'test'], batch_size=-1)

    x_train = (tfds.as_numpy(train_data))['image']
    y_train = (tfds.as_numpy(train_data))['label']
    x_test = (tfds.as_numpy(test_data))['image']
    y_test = (tfds.as_numpy(test_data))['label']
    
    #write_record(train_path, x_train, y_train)
    #write_record(test_path, x_test, y_test)

    train_image, train_label, test_image, test_label = read_record('D:/dataset/temp/', 'cifar10')
    '''

    timeit_num = 1
    tfds_time = timeit.timeit(lambda: data_loader.load_data('cifar10', load_mode='tfds'), number=timeit_num)
    tfrd_time = timeit.timeit(lambda: read_record('D:/dataset/temp/', 'cifar10'), number=timeit_num)
    path_time = timeit.timeit(lambda: data_loader.load_data('cifar10', load_mode='path'), number=timeit_num)

    print("tfds time:", tfds_time)
    print("tfrd time:", tfrd_time)
    print("path time:", path_time)
