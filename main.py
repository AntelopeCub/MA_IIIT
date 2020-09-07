import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import data_generator
import data_loader
import direction
import evaluation
import h5_util
import plot_1D
import plot_2D
from build_model import build_model
from quantization.build_vgg_qn import CUSTOM_OBJ


def main(model_type, 
         model_path, 
         batch_size = 128, 
         dataset    = 'cifar10', 
         load_mode  = 'tfrd',
         pre_mode   = 'norm',
         add_aug    = False, 
         aug_pol    = 'cifar_auto', 
         l2_reg_rate= None, 
         fc_type    = None,
         dir_path   = None, 
         fig_type   = '1D', 
         dot_num    = 11,
         l_range    = (-1, 1),
         loss_key   = 'train_loss',
         add_reg    = True,
        ):

    try:
        model = load_model(model_path)
    except Exception as e:
        model = build_model(model_type, dataset, fc_type=fc_type, l2_reg_rate=l2_reg_rate).model
        model.load_weights(model_path)
    
    if dir_path == None: 
        dir_path = model_path[:-3] + '_' + fig_type + '.h5'
    
    if os.path.exists(dir_path):
        print("Direction file is already created.")
        f = h5py.File(dir_path, 'r')
        
        if fig_type == '2D':
            assert ('xdirection' in f.keys() and 'ydirection' in f.keys()), "Please setup x/y direction!"
            set_y = True
        elif fig_type == '1D':
            assert 'xdirection' in f.keys(), "Please setup x direction!"
            set_y = False

        f.close()
    else:
        f = h5py.File(dir_path, 'w')

        xdirection = direction.creat_random_direction(model)
        h5_util.write_list(f, 'xdirection', xdirection)

        if fig_type == '2D':
            ydirection = direction.creat_random_direction(model)
            h5_util.write_list(f, 'ydirection', ydirection)
            set_y = True
        else:
            set_y = False
        
        f.close()
        print("Direction file created.")

    surf_path = dir_path[:-3] + '_surface' + '_' + str(dot_num) + '_' + loss_key + '_add_reg=' + str(add_reg) +'.h5'

    w = direction.get_weights(model)
    d = evaluation.load_directions(dir_path)

    evaluation.setup_surface_file(surf_path, dir_path, set_y, num=dot_num, l_range=l_range)

    if loss_key == 'train_loss':
        acc_key = 'train_acc'
        if not add_aug:
            x_set, y_set, _, _ = data_loader.load_data(dataset, load_mode=load_mode)
            x_mean = np.mean(x_set).astype('float32')
            x_std = np.std(x_set).astype('float32')
            #x_set = (x_set.astype('float32') - x_mean) / (x_std + 1e-7)
            x_set = data_generator.preprocess_input(x_set, x_mean, x_std, mode=pre_mode)
        else:
            print("Load temp dataset.")
            temp_file_path = data_generator.set_temp_dataset(dataset, load_mode, aug_pol, pre_mode=pre_mode)
            x_set, y_set = data_generator.load_temp_dataset(temp_file_path)
            print("Temp dataset loaded.")
            #if os.path.exists(temp_file_path):
            #    os.remove(temp_file_path)
            data_generator.remove_temp_dataset(temp_file_path)

    elif loss_key == 'test_loss':
        acc_key = 'test_acc'
        x_train, _, x_set, y_set= data_loader.load_data(dataset, load_mode=load_mode)
        x_mean = np.mean(x_train).astype('float32')
        x_std = np.std(x_train).astype('float32')
        #x_set = (x_set.astype('float32') - x_mean) / (x_std + 1e-7)
        x_set = data_generator.preprocess_input(x_set, x_mean, x_std, mode=pre_mode)

    else:
        raise Exception("Unknown loss key: %s" % (loss_key))

    evaluation.crunch(surf_path, model, model_type, w, d, x_set, y_set, loss_key, acc_key, batch_size=batch_size, add_reg=add_reg)

    if fig_type == '1D':
        plot_1D.plot_1d_loss_err(surf_path, xmin=l_range[0], xmax=l_range[1], loss_max=5, log=False, show=False)
    elif fig_type == '2D':
        plot_2D.plot_2d_contour(surf_path, surf_name=loss_key, vmin=0.1, vmax=10, vlevel=0.5, show=False)

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth while using cuda.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    

    tf.random.set_seed(123)

    model_type = 'vgg9_qn'
    model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_qn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_auto_199_0.9095_weights.h5"
    dataset = 'cifar10'
    fc_type = 'avg'
    load_mode = 'tfrd' if dataset == 'svhn_equal' else 'tfds'
    #pre_mode = 'norm'
    pre_mode = 'scale' if dataset == 'cifar10_pre' else 'norm'
    train_model = False
    l2_reg_rate = 5e-4
    batch_size = 128
    add_aug = False
    aug_pol = 'cifar_auto'
    plot_history = True
    workers = 1

    if train_model == True:
        model = build_model(model_type, dataset, l2_reg_rate=l2_reg_rate, fc_type=fc_type)
        model.train_model(batch_size=batch_size, load_mode=load_mode, add_aug=add_aug, aug_pol=aug_pol, plot_history=plot_history, workers=workers)
        model.model.save(model_path)

    fig_type = '2D'
    dot_num_list = [25, 51]
    l_range = (-1, 1)
    loss_key_list = ['train_loss', 'test_loss']
    add_reg = True
    
    for loss_key, dot_num in zip(loss_key_list, dot_num_list):
        main(model_type, model_path, batch_size=batch_size, dataset=dataset, load_mode=load_mode, 
             pre_mode=pre_mode, add_aug=add_aug, aug_pol=aug_pol, l2_reg_rate=l2_reg_rate, fc_type=fc_type,
             fig_type=fig_type, dot_num=dot_num, l_range=l_range, loss_key=loss_key, add_reg=add_reg)
