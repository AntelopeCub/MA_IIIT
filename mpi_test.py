import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import time

import h5py
import numpy as np
import tensorflow as tf

import data_generator
import data_loader
import direction
import evaluation
import h5_util
import mpi4tf as mpi
import plot_1D
import plot_2D
import scheduler
from build_model import build_model

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth while using cuda & mpi.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(123)

    dataset = 'svhn_equal'
    load_mode = 'tfrd'
    batch_size = 128
    aug_pol = 'svhn_base'
    temp_file_path = ''
    dot_num = 3
    set_y = True

    comm = mpi.setup_MPI()
    rank, nproc = comm.Get_rank(), comm.Get_size()

    model_path = "D:/Mitschke/Yanglin/MA_IIIT/models/resnet56/resnet56_128_norm_SGDNesterov_l2=0.0005_svhn_equal_077_0.9695_weights.h5"
    dir_path = "D:/Mitschke/Yanglin/MA_IIIT/models/resnet56/resnet56_128_norm_SGDNesterov_l2=0.0005_svhn_equal_077_0.9695_weights_2D_-0.2_0.2_same.h5"
    surf_path = "D:/Mitschke/Yanglin/MA_IIIT/models/resnet56/resnet56_128_norm_SGDNesterov_l2=0.0005_svhn_equal_077_0.9695_weights_2D_-0.2_0.2_same_mpi_test.h5"

    model = build_model('resnet56', dataset, fc_type='avg', l2_reg_rate=5e-4).model
    model.load_weights(model_path)
    w = direction.get_weights(model)
    d = evaluation.load_directions(dir_path)    

    if rank == 0:
        evaluation.setup_surface_file(surf_path, dir_path, set_y, num=dot_num, l_range=(-0.2, 0.2))
        print("Loading temp dataset.")
        sys.stdout.flush()
        temp_file_path = data_generator.set_temp_dataset(dataset, load_mode, aug_pol, pre_mode='norm')
        begin_time = time.time()
    else:
        temp_file_path = ''

    mpi.barrier(comm)

    temp_file_path = comm.bcast(temp_file_path, root=0)

    x_train, y_train = data_generator.load_temp_dataset(temp_file_path)
    print('Rank:%d loaded temp dataset' % (rank))
    sys.stdout.flush()
    
    '''
    #x_train, y_train, _, _ = data_loader.load_data(dataset, load_mode=load_mode)

    x_batch = x_train[batch_size*rank:batch_size*(rank+1)]
    y_batch = y_train[batch_size*rank:batch_size*(rank+1)]
    #x_batch = x_batch.astype('float32') / 255.0

    cce = tf.keras.losses.CategoricalCrossentropy()
    out = model(x_batch, training=False)
    loss = cce(y_batch, out).numpy()
    eq = tf.math.equal(tf.math.argmax(out, axis=1), tf.math.argmax(y_batch, axis=1))
    correct = np.sum(eq)
    acc = 1. * correct / batch_size

    print('Rank:%d, loss: %f, acc: %f' % (rank, loss, acc))
    sys.stdout.flush()
    '''

    f = h5py.File(surf_path, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    loss_key = 'train_loss'
    acc_key = 'train_acc'
    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)
    print('Computing %d values for rank %d'% (len(inds), rank))
    sys.stdout.flush()
    start_time = time.time()
    total_sync = 0.0

    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    for count, ind in enumerate(inds):
        coord = coords[count]
        evaluation.set_weights(model, w, d, coord)

        print('Rank:%d computing' % (rank))
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(model, cce, x_train, y_train, batch_size)
        loss_compute_time = time.time() - loss_start
    
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        syc_start = time.time()
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

        syc_time = time.time() - syc_start
        total_sync += syc_time

        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))
        sys.stdout.flush()
        
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)    

    total_time = time.time() - start_time
    print('Rank %d done! Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))
    sys.stdout.flush()

    f.close()
    
    mpi.barrier(comm)

    if rank == 0 and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        finish_time = time.time() - begin_time
        print("All rank finished, Total time: %.2f" % (finish_time))
