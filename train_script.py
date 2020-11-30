import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

import build_model

def Exp_LR_Scheduler(epoch, lr):
    return 0.1 * (0.5 ** (epoch // 20))

if __name__ == "__main__":
    
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth when using cuda.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    '''

    #tf.random.set_seed(123)

    model_type = 'vgg16_bn'
    dataset = 'cifar10'
    l2_reg_rate = 5e-4
    #l2_reg_rate = 1e-5
    fc_type = 'avg'
    load_mode = 'tfrd'
    train_model = True
    learning_rate = 0.1
    optimizer = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = Nadam()
    batch_size = 128
    add_aug = True
    aug_pol = 'svhn_auto'
    callbacks = []
    workers = 2
    epochs = 250
    pre_train = False
    pre_train_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/pretrain/vgg16_qn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_cifar_auto_010_0.4045_weights.h5"

    pre_mode = 'norm'
    #pre_mode = 'scale'
    #pre_mode = 'scale' if dataset == 'cifar10_pre' else 'norm'
    if 'vgg9' in model_type:
        model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/" + model_type + "_" + str(batch_size) + "_" + pre_mode + "_SGDNesterov_l2=" + str(l2_reg_rate) + "_" + fc_type + ".h5"
    elif 'vgg16' in model_type:
        model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/" + model_type + "_" + str(batch_size) + "_" + pre_mode + "_SGDNesterov_l2=" + str(l2_reg_rate) + "_" + fc_type + ".h5"
    elif 'resnet56' in model_type:
        model_path = "D:/Rain/text/Python/MA_IIIT/models/resnet56/" + model_type + "_" + str(batch_size) + "_" + pre_mode + "_SGDNesterov_l2=" + str(l2_reg_rate) + ".h5"

    if train_model == True:
        model = build_model.build_model(model_type, dataset, l2_reg_rate=l2_reg_rate, fc_type=fc_type, pre_mode=pre_mode)
        logs = ".\logs\log" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs, histogram_freq = 1, profile_batch = '5,10')
        #callbacks.append(tboard_callback)
        if add_aug:
            checkpoint_path = model_path[:-3] + '_' + dataset + '_' + aug_pol + "_{epoch:03d}_{val_accuracy:.4f}_weights.h5"
        else:
            checkpoint_path = model_path[:-3] + '_' + dataset + "_{epoch:03d}_{val_accuracy:.4f}_weights.h5"
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)
        callbacks.append(checkpoint_callback)
        lr_decay = LearningRateScheduler(Exp_LR_Scheduler) #for SGD#1
        #lr_decay = LearningRateScheduler(build_model.Cyclic_LR_Scheduler(0.002, epochs).cyc_decay) #for Nadam
        callbacks.append(lr_decay)
        if pre_train:
            model.model.load_weights(pre_train_path)
        model.train_model(optimizer=optimizer, batch_size=batch_size, epochs=epochs ,load_mode=load_mode, add_aug=add_aug, aug_pol=aug_pol, plot_history=True, callbacks=callbacks, workers=workers)
    else:
        weights_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9_qn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_004_0.3255_weights=7635.h5"
        model = build_model.build_model(model_type, dataset, l2_reg_rate=l2_reg_rate, fc_type=fc_type).model
        model.load_weights(weights_path)

    a = 1