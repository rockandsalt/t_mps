from os import path

import numpy as np
from scipy import ndimage as spim

from skimage.external import tifffile
from skimage import filters

def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion



class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
    
    def __init__(self, t_1, t_2, init = 1000):
        diff = np.logical_not(np.logical_xor(t_1,t_2))

        self._field = self.compute_potential_field(diff, end_pt_gain = 5.0)

    def generate_initial_point(self, smoothness = 5.0, trial = 1000):
        init_face = self._field[0,...]

        for size in  


    @staticmethod
    def compute_repulsive_field(im):
        obs = spim.morphology.distance_transform_edt(im)
        side = np.ones_like(im)

        

        return obs

    @staticmethod
    def compute_attractive_field(im, gain = 5.0):
        init = np.ones_like(im)
        init[0,...] = 0.0

        potential_field = -1.0*spim.morphology.distance_transform_edt(init)
        return 0.5*gain*potential_field

    @staticmethod
    def compute_potential_field(im, end_pt_gain = 5.0):
        attract = compute_attractive_field(im, end_pt_gain)
        repulsive = compute_repulsive_field(im)
        return attract + repulsive

    def potential_field_planning(self, im):
        pmap = compute_potential_field(im, end_pt_gain = 5.0)

        init = np.argmin(np.sum(pmap[0,...], axis = 1))
        wave = np.zeros((pmap.shape[1],2))
        wave[:,0] = init
        potential = pmap[0,...]

        motion = get_motion_model()

        for i in range(len(wave)):
            pos_x = wave[i][0]
            pos_y = wave[i][1]

            minp = np.inf

            s_map = pmap[i,...]

            minx, miny = -1, -1

            for j, _ in enumerate(motion):
                inx = int(pos_x + motion[j][0])
                iny = int(pos_y + motion[j][1])

                if inx >= s_map.shape[0] or iny >= s_map.shape[1]:
                    p = np.inf
                else:
                    p = pmap[inx][iny]
                if minp > p:
                    minp = p
                    minx = inx
                    miny = iny

            wave[i][0] = minx
            wave[i][1] = miny    

        

if __name__ == "__main__":
    im_path = path.join('.', 'data', 'CT_4_cr.tiff')

    image = tifffile.imread(im_path)

    otsu_threshold = filters.threshold_otsu(image)
    binary_im = image < otsu_threshold

    t_1 = binary_im[0:50,0:50,0:50]
    t_2 = binary_im[100:150,100:150,100:150]


