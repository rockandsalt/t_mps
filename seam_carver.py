# Adapted from andrew campbel implementation 
# https://github.com/andrewdcampbell/seam-carving 

from os import path
from skimage.external import tifffile
from skimage import filters

import numpy as np
import cv2
from numba import jit, njit, prange
from scipy import ndimage as ndi

from matplotlib import pyplot as plt


########################################
# ENERGY FUNCTIONS
########################################

def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h = im.shape[0]
    w = im.shape[1]

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
        
    return energy

def get_minimum_seam(im):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h = im.shape[0]
    w = im.shape[1]

    M = forward_energy(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask
        

def merge_image(t_1, t_2):
    diff = np.abs(t_1 - t_2)

    seam_idx, boolmask = get_minimum_seam(diff)

    labelled, num_f = ndi.label(boolmask.astype('int'))

    output = np.zeros_like(t_1)

    output[labelled == 1] = t_1[labelled == 1]
    output[labelled == 2] = t_2[labelled == 2]

    return output

if __name__ == "__main__":
    im_path = path.join('.', 'data', 'CT_4_cr.tiff')

    image = tifffile.imread(im_path)

    t_1 = image[0:150,0:150,25]
    t_2 = image[100:250,100:250,125]

    output = merge_image(t_1, t_2)

    fig, ax = plt.subplots()
    ax.imshow(output, cmap = 'gray')
    plt.show()
