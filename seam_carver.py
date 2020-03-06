# Adapted from andrew campbel implementation
# https://github.com/andrewdcampbell/seam-carving

from os import path
from skimage.external import tifffile
from skimage import filters
from skimage import exposure

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


def forward_energy_3D(im):
    h = im.shape[0]
    d = im.shape[2]

    energy = np.zeros_like(im)
    m = np.zeros_like(im)

    iU = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)

    kU = np.roll(im, 1, axis=2)

    cU = np.abs(R - L)
    cL = (np.abs(iU - L) + np.abs(kU - L))/2 + cU
    cR = (np.abs(iU - R) + np.abs(kU - R))/2 + cU

    for k in range(1, d):
        for i in range(1, h):
            mU = m[i-1,:,k-1]
            mL = np.roll(mU, 1)
            mR = np.roll(mU, -1)

            mULR = np.array([mU, mL, mR])
            cULR = np.array([cU[i,:,k], cL[i,:,k], cR[i,:,k]])

            mULR += cULR

            argmins = np.argmin(mULR, axis=0)
            m[i,:,k] = np.choose(argmins, mULR)
            energy[i, :, k] = np.choose(argmins, cULR)

    return energy


def get_minimum_seam(im):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h = im.shape[0]
    w = im.shape[1]
    d = im.shape[2]

    M = forward_energy_3D(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    for k in range(1, d):
        # populate DP matrix
        for i in range(1, h):
            for j in range(0, w):
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2, k - 1])
                    backtrack[i, j, k] = idx + j
                    min_energy = M[i-1, idx + j, k-1]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2, k - 1])
                    backtrack[i, j, k] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1, k - 1]

                M[i, j, k] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w, d), dtype=np.bool)
    j = np.argmin(M[-1, :, -1])
    for i in range(h-1, -1, -1):
        for k in range(d-1, -1, -1):
            boolmask[i, j, k] = False
            seam_idx.append((i, j, k))
            j = backtrack[i, j, k]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


def merge_image(t_1, t_2):
    diff = np.abs(t_1 - t_2)

    seam_idx, boolmask = get_minimum_seam(diff)

    return boolmask


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
    ax[0].imshow(t_1[50,...])
    ax[1].imshow(label[50,...])
    ax[2].imshow(t_2[50,...])
    plt.show()
