from os import path
from skimage.external import tifffile
from skimage import filters
from skimage import exposure

import numpy as np
import networkx as nx
from numba import jit, njit, prange
from scipy import ndimage as ndi

from matplotlib import pyplot as plt


def create_graph(im):
    shape = im.shape
    G = nx.DiGraph()

    ids = np.arange(len(im.ravel())).reshape(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # cost to remove ijk
                if(i - 1 > 0 and (i+1) < shape[0]):
                    G.add_edge(ids[i, j, k], ids[i + 1, j, k],
                            weight=np.abs(im[i+1, j, k] - im[i-1, j, k]))
                if(j - 1 > 0 and (j+1) < shape[1]):
                    G.add_edge(ids[i, j, k], ids[i, j + 1, k],
                            weight=np.abs(im[i, j+1, k] - im[i, j-1, k]))
                
                # +LU 
                G.add_edge(ids[i, j, k], ids[i, j, k - 1],
                           weight=np.abs(im[i, j, k-1] - im[i-1, j, k]) +
                           np.abs(im[i, j, k-1] - im[i, j-1, k]))
                # -LU 
                G.add_edge(ids[i, j, k], ids[i, j, k + 1],
                           weight=np.abs(im[i, j, k+1] - im[i-1, j, k]) +
                           np.abs(im[i, j, k-1] - im[i, j-1, k]))

    return ids, G

if __name__ == "__main__":
    im_path = path.join('.', 'data', 'CT_4_cr.tiff')

    image = tifffile.imread(im_path)
    image = exposure.equalize_hist(image)

    t_1 = image[0:150, 0:150, 0:150]
    t_2 = image[100:250, 100:250, 100:250]

    label = merge_image(t_1, t_2)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(t_1[..., 50])
    ax[1].imshow(label[..., 50])
    ax[2].imshow(t_2[..., 50])
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(t_1[50, ...])
    ax[1].imshow(label[50, ...])
    ax[2].imshow(t_2[50, ...])
    plt.show()
