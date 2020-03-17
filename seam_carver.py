from os import path
from skimage.external import tifffile
from skimage import filters
from skimage import exposure

import numpy as np

from graph_tool import Graph
from graph_tool import flow

from numba import jit, njit, prange
from scipy import ndimage as ndi

from matplotlib import pyplot as plt

from collections import Counter
from itertools import product


def create_graph(im):
    shape = im.shape
    G = Graph(directed=True)
    e_prop = g.new_edge_property("weight")

    ids = np.arange(len(im.ravel())).reshape(shape) + 1

    source = 0
    sink = np.max(ids) + 1

    iter_ijk = product(range(shape[0]),
                       range(shape[1]),
                       range(shape[2]))

    for i, j, k in iter_ijk:
                # cost to remove ijk
        if(j < (shape[0] - 1)):
            e = G.add_edge(ids[i, j, k], ids[i, j+1, k])
            e_prop[e] = np.abs(im[i, j+1, k] - im[i, j-1, k])

        if(i < (shape[0]-1)):
            # i-LU
            e = G.add_edge(ids[i, j, k], ids[i+1, j, k])
            e_prop[e] = np.abs(im[i+1, j, k] - im[i, j-1, k])   
        if(i > 0):
            # i+LU
            e = G.add_edge(ids[i, j, k], ids[i-1, j, k])
            e_prop[e] = np.abs(im[i-1, j, k] - im[i, j-1, k])
        if(k < (shape[2]-1)):
            # k-LU
            e = G.add_edge(ids[i, j, k], ids[i, j, k+1])
            e_prop[e] = np.abs(im[i, j, k+1] - im[i, j-1, k])
        if(k > 0):
            # k+LU
            e = G.add_edge(ids[i, j, k], ids[i, j, k-1])
            e_prop[e] = np.abs(im[i, j, k-1] - im[i, j-1, k])

        if j == 0:
            e = G.add_edge(source, ids[i, j, k])
            e_prop[e] = 1.0

        if j == (shape[2] - 1):
            e = G.add_edge(ids[i, j, k], sink)
            e_prop[e] = 1.0

    return G, ids, source, sink


def merge_image(t_1, t_2):
    diff = np.abs(t_1 - t_2)

    graph, ids, source, sink = create_graph(diff)
    cap = g.edge_properties["weight"]
    res = flow.boykov_kolmogorov_max_flow(graph, source, sink, capacity = cap)
    partition = flow.min_st_cut(graph, source, cap, res)
    
    output = np.zeros_like(t_1)

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
