import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import copy

import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

import h5_util

def creat_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    norm_direction = normalize_directions_for_weights(direction, weights)
    return norm_direction

def get_weights(model):
    return [tf.convert_to_tensor(p) for p in model.weights]

def get_random_weights(weights):
    return [tf.random.normal(w.shape) for w in weights]

def normalize_directions_for_weights(direction, weights):
    assert(len(direction) == len(weights))
    return [(d * tf.norm(w) / (tf.norm(d) + 1e-10)) for d, w in zip(direction, weights)]

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU') #must limit gpu memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15.h5"

    model = load_model(model_path)

    dir_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/directions/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15_weights_2D.h5"

    f = h5py.File(dir_path, 'w')

    tf.random.set_seed(123)

    set_y = True

    xdirection = creat_random_direction(model)
    h5_util.write_list(f, 'xdirection', xdirection)

    if set_y:
        ydirection = creat_random_direction(model)
        h5_util.write_list(f, 'ydirection', ydirection)

    f.close()
    print("direction file created")

    a = 1
