import numpy as np
from scipy import signal
import skimage as ski
from skimage.external import tifffile
from skimage import filters
from skimage import feature
from tqdm import tqdm
from scipy import ndimage as spim
from sklearn.neighbors import BallTree

from os import path

from matplotlib import pyplot as plt

class t_mps():
    def __init__(self, DI, t_size, overlap = 0.30, recon_size = (2,2,2)):
        super().__init__()

        self._di = DI
        self._t_size = t_size
        self._overlap = int(t_size*overlap)

        self._recon_grid = np.zeros(recon_size).astype('int')
        self._template_list = []

    def get_template( self,center):
        template = self._di[
            int(center[0] - int(t_size)/2): int(center[0] + int(t_size)/2),
            int(center[1] - int(t_size)/2): int(center[1] + int(t_size)/2),
            int(center[2] - int(t_size)/2): int(center[2] + int(t_size)/2)
        ]

        return template
    
    def filter_at_boundary(self, peaks):
        filter_x = np.logical_and(int(self._t_size/2) < peaks[:,0],peaks[:,0]  < self._di.shape[0] - int(self._t_size/2) )
        filter_y = np.logical_and(int(self._t_size/2) < peaks[:,1],peaks[:,1]  < self._di.shape[1] - int(self._t_size/2) )
        filter_z = np.logical_and(int(self._t_size/2) < peaks[:,2],peaks[:,2]  < self._di.shape[2] - int(self._t_size/2) )

        overall = (filter_x*filter_y*filter_z).astype('bool')

        return peaks[overall,...]

    def find_template(self, i, j, k):

        list_ols = []

        if(self._recon_grid[i,j,k-1]):
            id = self._recon_grid[i,j,k-1] - 1
            template = self._template_list[id]
            ols = template[:,:,-self._overlap:]

            list_ols.append(ols)
        
        if(self._recon_grid[i,j-1,k]):
            id = self._recon_grid[i,j-1,k] - 1
            template = self._template_list[id]
            ols = template[:,-self._overlap:,:]

            list_ols.append(ols)
        
        if(self._recon_grid[i-1,j,k]):
            id = self._recon_grid[i-1,j,k] - 1
            template = self._template_list[id]
            ols = template[-self._overlap:,:,:]

            list_ols.append(ols)
        
        convo_res = np.zeros_like(self._di)
        for ols in list_ols:
            convo_res += signal.convolve(self._di, ols, mode='same')
        
        peaks = feature.peak_local_max(
                        convo_res, min_distance=int(t_size/2), threshold_rel=0.70, 
                        indices=True)

        filtered_peak = self.filter_at_boundary(peaks)
                    
        peak_id = np.random.choice(np.arange(0,filtered_peak.shape[0]))

        template = self.get_template(filtered_peak[peak_id])

        self._template_list.append(template)
        self._recon_grid[i,j,k] = len(self._template_list)
    
    def smooth_discontinuity(self, t_1, t_2):
        diff = np.logical_not(np.logical_xor(t_1,t_2))

    
    def _build_image(self):
        t_size = self._t_size
        im = np.zeros(np.array(self._recon_grid.shape)*self._t_size)

        for i in range(0,self._recon_grid.shape[0]):
            for j in range(0,self._recon_grid.shape[1]):
                for k in range(0,self._recon_grid.shape[2]):
                    id_template = self._recon_grid[i,j,k]
                    template = self._template_list[id_template - 1]

                    im[
                        i*t_size:(i+1)*t_size,
                        j*t_size:(j+1)*t_size,
                        k*t_size:(k+1)*t_size
                    ] = template

        return im

    def generate(self, num_recon=1):
        
        init_coord = [np.random.randint(self._di.shape[i]) for i in range(3)]

        template = self._di[init_coord[0]:init_coord[0] + t_size,
                    init_coord[1]:init_coord[1] + t_size,
                    init_coord[2]:init_coord[2] + t_size]

        self._recon_grid[0,0,0] = 1
        self._template_list.append(template)
        with tqdm(total = np.prod(self._recon_grid.shape)) as pbar:
            for i in np.arange(0,self._recon_grid.shape[0]):
                for j in np.arange(0,self._recon_grid.shape[1]):
                    for k in np.arange(0,self._recon_grid.shape[2]):

                        if(i != 0 or j != 0 or k!= 0):
                            self.find_template(i,j,k)
                            pbar.update(1)

        return self._build_image()

if __name__ == "__main__":
    im_path = path.join('.', 'data', 'CT_4_cr.tiff')

    image = tifffile.imread(im_path)

    otsu_threshold = filters.threshold_otsu(image)
    binary_im = image < otsu_threshold

    t_size = 149

    recon = t_mps(binary_im.astype('float'), t_size)
    recon_im = recon.generate()

    fig, ax = plt.subplots()
    ax.imshow(recon_im[:,:,10], cmap = 'gray')
    plt.show()
