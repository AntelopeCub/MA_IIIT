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


def main(model_type, 
         model_path, 
         batch_size=128, 
         dataset='cifar10', 
         load_mode='tfds',
         add_aug=False, 
         aug_pol='cifar_auto', 
         l2_reg_rate=None, 
         fc_type=None,
         dir_path=None, 
         fig_type='1D', 
         dot_num=11,
         loss_key='train_loss' 
        ):
    
    try:
        model = load_model(model_path)
    except Exception as e:
        model = build_model(model_type, dataset, fc_type=fc_type, l2_reg_rate=l2_reg_rate).model
        model.load_weights(model_path)
    
    if dir_path == None: 
        dir_path = model_path[:-3] + '_' + fig_type + '.h5'
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
    else:
        if os.path.exists(dir_path):
            f = h5py.File(dir_path, 'r')
            
            if fig_type == '2D':
                assert ('xdirection' in f.keys() and 'ydirection' in f.keys()), "Please set up x/y direction!"
                set_y = True
            elif fig_type == '1D':
                assert 'xdirection' in f.keys(), "Please set up x direction!"
                set_y = False

            f.close()
        else:
            raise Exception("Direction file doesn't exist!")

    surf_path = dir_path[:-3] + '_surface' + '_' + str(dot_num) + '.h5'

    w = direction.get_weights(model)
    d = evaluation.load_directions(dir_path)

    evaluation.setup_surface_file(surf_path, dir_path, set_y, num=dot_num)

    if loss_key == 'train_loss':
        acc_key = 'train_acc'
        if not add_aug:
            x_set, y_set, _, _ = data_loader.load_data(dataset, load_mode=load_mode)
            x_mean = np.mean(x_set).astype('float32')
            x_std = np.std(x_set).astype('float32')
            x_set = (x_set.astype('float32') - x_mean) / (x_std + 1e-7)
        else:
            print("Load temp dataset.")
            temp_file_path = data_generator.set_temp_dataset(dataset, load_mode, aug_pol)
            x_set, y_set = data_generator.load_temp_dataset(temp_file_path)
            print("Temp dataset loaded.")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    elif loss_key == 'test_loss':
        acc_key = 'test_acc'
        x_train, _, x_set, y_set= data_loader.load_data(dataset, load_mode=load_mode)
        x_mean = np.mean(x_train).astype('float32')
        x_std = np.std(x_train).astype('float32')
        x_set = (x_set.astype('float32') - x_mean) / (x_std + 1e-7)

    else:
        raise Exception("Unknown loss key: %s" % (loss_key))

    evaluation.crunch(surf_path, model, w, d, x_set, y_set, loss_key, acc_key, batch_size)

    if fig_type == '1D':
        plot_1D.plot_1d_loss_err(surf_path, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False)
    elif fig_type == '2D':
        plot_2D.plot_2d_contour(surf_path, surf_name=loss_key, vmin=0.1, vmax=10, vlevel=0.5, show=False)

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth while using cuda.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    

    tf.random.set_seed(123)

    model_type = 'vgg16_bn'
    model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_baseline_216_0.9410_weights.h5"
    dataset = 'cifar10'
    fc_type = 'avg'
    load_mode = 'tfds'
    train_model = False
    l2_reg_rate = 5e-4
    batch_size = 128
    add_aug = True
    aug_pol = 'baseline'
    plot_history = True
    workers = 1

    if train_model == True:
        model = build_model(model_type, dataset, l2_reg_rate=l2_reg_rate, fc_type=fc_type)
        model.train_model(batch_size=batch_size, load_mode=load_mode, add_aug=add_aug, aug_pol=aug_pol, plot_history=plot_history, workers=workers)
        model.model.save(model_path)

    fig_type = '2D'
    dot_num = 25
    loss_key = 'test_loss'
    
    main(model_type, model_path, batch_size=batch_size, dataset=dataset, load_mode=load_mode, 
         add_aug=add_aug, aug_pol=aug_pol, l2_reg_rate=l2_reg_rate, fc_type=fc_type,
         fig_type=fig_type, dot_num=dot_num, loss_key=loss_key)
