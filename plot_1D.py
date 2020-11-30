import h5py
import matplotlib.pyplot as plt

import h5_util


def plot_1d_loss_err(surf_path, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_path, 'r')
    x = f['xcoordinates'][:]
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    # loss and accuracy map
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left = 0.07, bottom = 0.06, right = 0.9, top = 0.98)
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
        ax1.set_ylim(1e-1, 1e1)
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        ax1.set_ylim(0, loss_max)
    tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=1)

    plt.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 1)
    plt.savefig(surf_path + '_1d_loss_acc' + ('_log' if log else '') + '.pdf', dpi=300, format='pdf')

    if show: 
        plt.show()
    f.close()   

def plot_1d_loss_diff(train_loss_path, test_loss_path, xmin=-1.0, xmax=2.0, loss_max=15, log=False, show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    with h5py.File(train_loss_path, 'r') as f:
        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]

    with h5py.File(test_loss_path, 'r') as f:
        assert (x == f['xcoordinates'][:]).all(), "X coordinates do not match!"
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 2.0 else max(x)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left = 0.09, bottom = 0.06, right = 0.9, top = 0.98)
    ax2 = ax1.twinx()

    if log:
        ax1.semilogy(x, train_loss, 'b-', label='Train loss', linewidth=1)
        ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        ax1.set_ylim(1e-1, 1e1)
    else:
        #ax1.plot(x, train_loss, 'b-', label='Train loss', linewidth=1)
        tel = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        ax1.set_ylim(0, loss_max)
    
    #ax2.plot(x, train_acc, 'r-', label='Train acc', linewidth=1)
    tea = ax2.plot(x, test_acc, 'r--', label='Test acc', linewidth=1)

    plt.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 1)

    te = tel + tea
    te_labels = [l.get_label() for l in te]
    ax1.legend(te, te_labels)

    if show: 
        plt.show()


if __name__ == "__main__":
    #surf_path = "C:/Users/Rain/Desktop/surface/diff/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_base_180_0.9472_weights_1D_-1.0_2.0_diff3_surface_3001_train_loss_add_reg=False.h5"
    #plot_1d_loss_err(surf_path, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=True)

    train_loss_path = 'C:/Users/Rain/Desktop/surface/diff/vgg16_qn_128_scale_Nadam_l2=0.00001_avg_cifar10_cifar_base_8bits_weights_1D_-1.0_2.0_diff3_surface_601_train_loss.h5'
    test_loss_path  = 'C:/Users/Rain/Desktop/surface/diff/vgg16_qn_128_scale_Nadam_l2=0.00001_avg_cifar10_cifar_base_8bits_weights_1D_-1.0_2.0_diff3_surface_601_test_loss.h5'
    plot_1d_loss_diff(train_loss_path, test_loss_path, show=True)