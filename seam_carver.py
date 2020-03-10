from os import path
from skimage.external import tifffile
from skimage import filters
from skimage import exposure

import numpy as np
import networkx as nx
from numba import jit, njit, prange
from scipy import ndimage as ndi

from matplotlib import pyplot as plt

from collections import Counter
from itertools import product


def create_graph(im):
    shape = im.shape
    G = nx.DiGraph()

    ids = np.arange(len(im.ravel())).reshape(shape) + 1

    source = 0
    sink = np.max(ids) + 1

    iter_ijk = product(range(shape[0]),
                       range(shape[1]),
                       range(shape[2]))

    for i, j, k in iter_ijk:
                # cost to remove ijk
        if(j < (shape[0] - 1)):
            G.add_edge(ids[i, j, k], ids[i, j+1, k],
                       capacity=np.abs(im[i, j+1, k] - im[i, j-1, k]))

        if(i < (shape[0]-1)):
            # i-LU
            G.add_edge(ids[i, j, k], ids[i+1, j, k],
                       capacity=np.abs(im[i+1, j, k] - im[i, j-1, k]))
        if(i > 0):
            # i+LU
            G.add_edge(ids[i, j, k], ids[i-1, j, k],
                       capacity=np.abs(im[i-1, j, k] - im[i, j-1, k]))

        if(k < (shape[2]-1)):
            # k-LU
            G.add_edge(ids[i, j, k], ids[i, j, k+1],
                       capacity=np.abs(im[i, j, k+1] - im[i, j-1, k]))

        if(k > 0):
            # k+LU
            G.add_edge(ids[i, j, k], ids[i, j, k-1],
                       capacity=np.abs(im[i, j, k-1] - im[i, j-1, k]))

        if j == 0:
            G.add_edge(source, ids[i, j, k], capacity=1.0)

        if j == (shape[2] - 1):
            G.add_edge(ids[i, j, k], sink, capacity=1.0)

    return G, ids, source, sink


def merge_image(t_1, t_2):
    diff = np.abs(t_1 - t_2)

    graph, ids, source, sink = create_graph(diff)

    cut, partition = nx.minimum_cut(graph, source, sink)

    t_1_set, t_2_set = partition
    output = np.zeros_like(t_1)

    print(Counter(t_1_set) == Counter(t_1_set))

    for node in t_1_set:
        if node != source and node != sink:
            indice = np.nonzero(ids == node)
            output[indice] = t_1[indice]

    for node in t_2_set:
        if node != source and node != sink:
            indice = np.nonzero(ids == node)
            output[indice] = t_2[indice]

    return output


if __name__ == "__main__":
    im_path = path.join('.', 'data', 'CT_4_cr.tiff')

    image = tifffile.imread(im_path)
    image = exposure.equalize_hist(image)

    t_1 = image[0:150, 0:150, 0:150]
    t_2 = image[250:400, 250:400, 250:400]

    output = merge_image(t_1, t_2)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(t_1[..., 50])
    axs[1].imshow(output[..., 50])
    axs[2].imshow(t_2[..., 50])
    plt.show()
