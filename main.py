import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import h5py
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


def main(model_type, model_path, batch_size, dataset, load_mode, fig_type, dot_num=11, add_aug=False, aug_pol='baseline'):
    
    try:
        model = load_model(model_path)
    except Exception as e:
        model = build_model(model_type, dataset).model
        model.load_weights(model_path)
    
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
    print("direction file created")

    surf_path = dir_path[:-3] + '_surface.h5'

    w = direction.get_weights(model)
    d = evaluation.load_directions(dir_path)

    evaluation.setup_surface_file(surf_path, dir_path, set_y, num=dot_num)

    if not add_aug:
        x_train, y_train, _, _ = data_loader.load_data(dataset, load_mode=load_mode)
        x_train = x_train / 255.0
    else:
        print("Load temp dataset.")
        temp_file_path = data_generator.set_temp_dataset(dataset, load_mode, aug_pol)
        x_train, y_train = data_generator.load_temp_dataset(temp_file_path)
        print("Temp dataset loaded.")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    evaluation.crunch(surf_path, model, w, d, x_train, y_train, 'train_loss', 'train_acc', batch_size)

    if fig_type == '1D':
        plot_1D.plot_1d_loss_err(surf_path, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=True)
    elif fig_type == '2D':
        plot_2D.plot_2d_contour(surf_path, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=True)

if __name__ == "__main__":
    
    
    gpus = tf.config.experimental.list_physical_devices('GPU') #should limit gpu memory growth while using cuda.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    

    tf.random.set_seed(123)

    model_type = 'vgg16_bn'
    model_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/vgg16_bn_cifar_pol_42_0.8499_weights.h5"
    dataset = 'cifar10'
    load_mode = 'tfds'
    train_model = False
    batch_size = 128
    add_aug = True
    aug_pol = 'cifar_pol'
    plot_history = True
    workers = 1

    if train_model == True:
        model = build_model(model_type, dataset)
        model.train_model(batch_size=batch_size, load_mode=load_mode, add_aug=add_aug, aug_pol=aug_pol, plot_history=plot_history, workers=workers)
        model.model.save(model_path)

    fig_type = '2D'
    dot_num = 21
    
    main(model_type, model_path, batch_size, dataset, load_mode, fig_type, dot_num=dot_num, add_aug=add_aug, aug_pol=aug_pol)
