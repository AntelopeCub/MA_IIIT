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
    fig.savefig(surf_path + '_' + surf_name + '_3dsurface.pdf', dpi=300, format='pdf')

    f.close()

    if show: 
        plt.show()


if __name__ == "__main__":

    surf_path = "D:/Rain/text/Python/MA_IIIT/models/vgg16/surface/vgg16_bn_cifar_pol_42_0.8499_weights_2D_surface.h5"

    #plot_2d_contour(surf_path, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=True)
    plot_3d_surface(surf_path, surf_name='train_loss', show=True)
