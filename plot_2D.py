import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_contour(surf_path, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    f = h5py.File(surf_path, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')

    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_path + '_' + surf_name + '_2dcontour.pdf', dpi=300, format='pdf')

    f.close()

    if show:
        plt.show()
    

def  plot_3d_surface(surf_path, surf_name='train_loss', show=False):
    f = h5py.File(surf_path, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    
    print('------------------------------------------------------------------')
    print('plot_3d_surface')
    print('------------------------------------------------------------------')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #fig.savefig(surf_path + '_' + surf_name + '_3dsurface.pdf', dpi=300, format='pdf')

    f.close()

    if show: 
        plt.show()


def set_surface_zeros(surf_path_list, surf_name='train_loss'):
    Z_list = []

    for surf_path in surf_path_list:
        f = h5py.File(surf_path, 'r')
        if surf_name in f.keys():
            Z_list.append(np.array(f[surf_name][:]))
        else:
            raise Exception('%s is not in surface file: %s' % (surf_name, surf_path))
        f.close()
    
    Z_list = np.asarray(Z_list)
    Z_min = np.min(Z_list)
    
    Z_list = Z_list - Z_min
    for Z, surf_path in zip(Z_list, surf_path_list):
        f = h5py.File(surf_path, 'r+')
        
        if surf_name+'_zeros' in f.keys():
            f[surf_name+'_zeros'][:] = Z.tolist()
        else:
            f[surf_name+'_zeros'] = Z.tolist()

        f.flush()
        f.close()
    

if __name__ == "__main__":
    
    '''
    surf_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/surface/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_auto_247_0.9507_weights_same_surface_51_test_loss.h5"

    plot_2d_contour(surf_path, surf_name='test_acc', vmin=0.1, vmax=15, vlevel=0.2, show=True)
    #plot_3d_surface(surf_path, surf_name='train_loss', show=True)
    '''

    surf_path_list = [
        'D:/Rain/text/Python/MA_IIIT/models/vgg9/surface/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_pre_127_0.9105_weights_-1_1_same_surface_25_train_loss.h5',
        'D:/Rain/text/Python/MA_IIIT/models/vgg9/surface/vgg9_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar10_pre_cifar_base_195_0.9451_weights_-1_1_same_surface_25_train_loss.h5',
    ]

    set_surface_zeros(surf_path_list=surf_path_list, surf_name='train_loss')