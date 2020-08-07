import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import math
import time
import timeit

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import data_loader

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
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
        x_split = x_set[split_size*i:min(split_size*(i+1), len(x_set))]
        y_split = y_set[split_size*i:min(split_size*(i+1), len(y_set))]
        with tf.io.TFRecordWriter(file_path_split) as file_writer:
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_split.tobytes()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=y_split)),
                "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(x_split.shape))),
                "imgdt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(x_split.dtype).encode()])),
                "labdt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(y_split.dtype).encode()])),
            })).SerializeToString()
            file_writer.write(record_bytes)

def read_record(file_path, dataset):
    assert os.path.exists(file_path), 'Dataset does not exsist, please check path: ' + file_path
    train_files = glob.glob(file_path + dataset + '-train.tfrecord*')
    test_files = glob.glob(file_path + dataset + '-test.tfrecord*')

    assert (len(train_files) > 0 and len(test_files) > 0), 'Train/Test data is missing, please check!'
    assert (int(train_files[0].split('-')[-1]) == len(train_files) and int(test_files[0].split('-')[-1]) == len(test_files)), 'Several Train/Test records are missing, please check!'

    train_image, train_label = extract_record(train_files)
    test_image, test_label = extract_record(test_files)

    return train_image, train_label, test_image, test_label

def extract_record(file_list):
    image_list = None
    label_list = None
    for record_file in file_list:
        raw_record = tf.data.TFRecordDataset(record_file)
        parsed_record = raw_record.map(_parse_function)
        for features in parsed_record:
            imgdt = features['imgdt'].numpy().decode()
            labdt = features['labdt'].numpy().decode()
            image = np.frombuffer(features['image'].numpy(), dtype=imgdt).reshape(features['shape'].numpy())
            label = features['label'].numpy().astype(labdt)
            image_list = image if image_list is None else np.concatenate((image_list, image), axis=0)
            label_list = label if label_list is None else np.concatenate((label_list, label), axis=0)
    
    return image_list, label_list



if __name__ == "__main__":

    dataset = 'svhn_equal'

    '''
    train_path = os.path.join('D:/dataset/temp/', dataset + "-train.tfrecord")
    test_path = os.path.join('D:/dataset/temp/', dataset + "-test.tfrecord")
    
    start = time.time()
    #train_data, test_data = tfds.load(name='cifar10', split=['train', 'test'], batch_size=-1)
    x_train, y_train, x_test, y_test = data_loader.load_data(dataset, load_mode='path')

    y_train = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train
    y_test = np.argmax(y_test, axis=1) if y_test.ndim == 2 else y_test
    path_time = time.time() - start
    print("path time:", path_time)

    write_record(train_path, x_train, y_train)
    write_record(test_path, x_test, y_test)
    '''

    start = time.time()
    train_image, train_label, test_image, test_label = read_record('D:/dataset/tfrd/', dataset)
    tfrd_time = time.time() - start

    timeit_num = 1
    
    #tfds_time = timeit.timeit(lambda: data_loader.load_data('cifar10', load_mode='tfds'), number=timeit_num)
    #tfrd_time = timeit.timeit(lambda: read_record('D:/dataset/temp/', 'cifar10'), number=timeit_num)
    #path_time = timeit.timeit(lambda: data_loader.load_data('cifar10', load_mode='path'), number=timeit_num)
    
    #print("tfds time:", tfds_time)
    #print("path time:", path_time)
    print("tfrd time:", tfrd_time)
    
    #a, b, c = tfds.load(name='svhn_cropped', split=['train', 'test', 'extra'], batch_size=-1)
    #tfds_time = timeit.timeit(lambda: tfds.load(name='svhn_cropped', split=['train', 'test', 'extra'], batch_size=-1), number=timeit_num)
    #print("tfds time:", tfds_time)
    a = 1
