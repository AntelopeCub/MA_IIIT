import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
import numpy as np
import tensorflow as tf

import direction
import evaluation
import h5_util
from build_model import build_model
from main import main
from quantization.build_vgg_qn import CUSTOM_OBJ


if __name__ == "__main__":
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth while using cuda.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    '''
    
    #tf.random.set_seed(123)

    #model_type = 'vgg16_qn'
    model_type_list = ['vgg16_bn']
    L_A = [3, 5] #[3, 5]
    L_W = [1, 7] #[1, 7]
    dataset = 'cifar10'
    fc_type = 'avg'
    load_mode = 'tfrd'
    #pre_mode = 'norm'
    #pre_mode = 'scale'
    #l2_reg_rate = 1e-5 if 'qn' in model_type else 5e-4
    batch_size = 128

    run_dict = {
        "vgg9_bn": [            
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_243_0.9123_weights.h5",
                "add_aug": False,
                "aug_pol": "cifar_base"
            },
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_base_180_0.9472_weights.h5",
                "add_aug": True,
                "aug_pol": "cifar_base"
            },
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg9/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_auto_246_0.9526_weights.h5",
                "add_aug": True,
                "aug_pol": "cifar_auto"
            },
        ],
        "vgg16_bn": [
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg16/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_171_0.8944_weights.h5",
                "add_aug": False,
                "aug_pol": "cifar_base"
            },
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg16/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_base_212_0.9486_weights.h5",
                "add_aug": True,
                "aug_pol": "cifar_base"
            },
            {
                "model_path": "D:/Rain/text/Python/MA_IIIT/models/vgg16/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_auto_237_0.9561_weights.h5",
                "add_aug": True,
                "aug_pol": "cifar_auto"
            },
        ],
              
    }

    fig_type = '2D'
    dot_num_list = [3, 3]
    l_range = (-0.03, 0.03)
    loss_key_list = ['train_loss']
    add_reg = False
    
    for model_type in model_type_list:
        l2_reg_rate = 1e-5 if 'qn' in model_type else 5e-4
        model = build_model(model_type, dataset, fc_type=fc_type, l2_reg_rate=l2_reg_rate, L_A=L_A, L_W=L_W).model
    
        w = direction.get_weights(model)
        dx = direction.get_random_weights(w)
        dy = direction.get_random_weights(w)

        run_list = run_dict[model_type]
        for run in run_list:
            model_path = run["model_path"]
            add_aug = run["add_aug"]
            aug_pol = run["aug_pol"]

            model.load_weights(model_path)
            w = direction.get_weights(model)
            norm_dx = direction.normalize_directions_for_weights(dx, w)
            norm_dy = direction.normalize_directions_for_weights(dy, w)

            dir_path = model_path[:-3] + '_' + fig_type + '_' + str(l_range[0]) + '_' + str(l_range[1]) + '_same.h5'
            f = h5py.File(dir_path, 'w')
            h5_util.write_list(f, 'xdirection', norm_dx)
            h5_util.write_list(f, 'ydirection', norm_dy)
            f.close()

            if 'norm' in model_path:
                pre_mode = 'norm'
            elif 'scale' in model_path:
                pre_mode = 'scale'
            else:
                raise Exception('Unknown pre_mode!')

            for loss_key, dot_num in zip(loss_key_list, dot_num_list):
                main(model_type, model_path, batch_size=batch_size, dataset=dataset, load_mode=load_mode, L_A=L_A, L_W=L_W,
                    pre_mode=pre_mode, add_aug=add_aug, aug_pol=aug_pol, l2_reg_rate=l2_reg_rate, fc_type=fc_type,
                    dir_path=dir_path, fig_type=fig_type, dot_num=dot_num, l_range=l_range, loss_key=loss_key, add_reg=add_reg)
