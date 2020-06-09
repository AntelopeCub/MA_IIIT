import h5py
import matplotlib.pyplot as plt
import numpy as np

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
    fig.savefig(surf_path + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300, format='pdf')

    if show:
        plt.show()
    f.close()

if __name__ == "__main__":

    surf_path = "D:/Rain/text/Python/MA_IIIT/models/vgg9/surface/vgg9_sgd_lr=0.1_bs=128_wd=0.0_epochs=15_surface_2D.h5"

    plot_2d_contour(surf_path, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=True)
