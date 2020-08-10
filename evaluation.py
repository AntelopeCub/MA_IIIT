import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import sys
import time

import h5py
import numpy as np
import tensorflow as tf

import data_loader
import direction
import h5_util


def setup_surface_file(surf_path, dir_path, set_y, num=51, l_range=(-1, 1)):
    l_min = l_range[0]
    l_max = l_range[1]
    assert l_min < l_max, 'Invalid range: ' + str(l_range)

    f = h5py.File(surf_path, 'a')
    f['dir_path'] = dir_path

    xcoordinates = np.linspace(l_min, l_max, num=num)
    f['xcoordinates'] = xcoordinates

    if set_y:
        ycoordinates = np.linspace(l_min, l_max, num=num)
        f['ycoordinates'] = ycoordinates

    f.close()

def load_directions(dir_path):
    f = h5py.File(dir_path, 'r')

    xdirections_data = h5_util.read_list(f, 'xdirection')
    if 'ydirection' in f.keys():
        ydirections_data = h5_util.read_list(f, 'ydirection')
        xdirections = [tf.convert_to_tensor(xdata) for xdata in xdirections_data]
        ydirections = [tf.convert_to_tensor(ydata) for ydata in ydirections_data]
        directions = [xdirections, ydirections]
    else:
        directions = [[tf.convert_to_tensor(xdata) for xdata in xdirections_data]]

    f.close()
    return directions

def set_weights(model, weights, directions=None, step=None):

    if len(directions) == 2:
        dx = directions[0]
        dy = directions[1]
        changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
    else:
        changes = [d*step for d in directions[0]]

    for idx in range(len(weights)):
        model.weights[idx].assign(weights[idx] + tf.convert_to_tensor(changes[idx]))

def eval_loss(model, cce, x_set, y_set, batch_size):
    total = len(x_set)
    step_num = math.ceil(total / batch_size)
    total_loss = 0
    reg_loss = 0
    correct = 0

    if len(model.losses) > 0:
        reg_loss = np.sum(model.losses)

    for idx in range(step_num):
        x = x_set[batch_size*idx:batch_size*(idx+1)]
        y = y_set[batch_size*idx:batch_size*(idx+1)]
        out = model(x, training=False)
        total_loss += cce(y, out).numpy()
        eq = tf.math.equal(tf.math.argmax(out, axis=1), tf.math.argmax(y, axis=1))
        correct += np.sum(eq)
    loss = total_loss / total + reg_loss
    acc = 1.*correct/total
    #print('loss: %f, acc: %f' % (loss, acc))
    #sys.stdout.flush()
    return loss, acc

def crunch(surf_path, model, w, d, x_set, y_set, loss_key, acc_key, batch_size=128):
    
    f = h5py.File(surf_path, 'r+')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        f[loss_key] = losses
        f[acc_key] = accuracies

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    start_time = time.time()

    if ycoordinates is not None:
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()
        s2 = ycoord_mesh.ravel()
        for idx, coord in enumerate(np.c_[s1,s2]):
            set_weights(model, w, d, coord)
            loss, acc = eval_loss(model, cce, x_set, y_set, batch_size)
            losses.ravel()[idx] = loss
            accuracies.ravel()[idx] = acc

            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

            print('coord=%s, \tloss: %f, acc: %f' % (str(coord), loss, acc))
            sys.stdout.flush()

    else:
        for idx, coord in enumerate(xcoordinates):
            set_weights(model, w, d, coord)
            loss, acc = eval_loss(model, cce, x_set, y_set, batch_size)
            losses.ravel()[idx] = loss
            accuracies.ravel()[idx] = acc

            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

            print('coord=%s, \tloss: %f, acc: %f' % (str(coord), loss, acc))
            sys.stdout.flush()

    f.close()
    total_time = time.time() - start_time
    print('Finished! Total time:%.2fs' % total_time)


if __name__ == "__main__":
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU') #must limit gpu memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    '''
    model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15.h5"
    dir_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/directions/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15_weights.h5"
    surf_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/surface/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15_surface_1D.h5"
    batch_size = 128

    model = tf.keras.models.load_model(model_path)
    w = direction.get_weights(model)
    d = load_directions(dir_path)

    set_y = False

    setup_surface_file(surf_path, dir_path, set_y, num=51)

    x_set, y_set, _, _ = data_loader.load_data('cifar10', load_mode='tfds')

    crunch(surf_path, model, w, d, x_set, y_set, 'train_loss', 'train_acc', batch_size)
