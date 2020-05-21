''' I/O utilties based on Fitsio
    MR: probably will shift to astropy.io.pyfits
'''


import numpy
import matplotlib.pyplot as plt

from astropy.io import fits


class array(numpy.ndarray):
    ''' Wrapper around numpy.ndarray to facilitate visualization of 2D images
    using imshow
    '''
    def __new__(cls, a):
        obj = numpy.asarray(a).view(cls)
        return obj

    def imshow(self, ax=None, vmin=None, vmax=None, origin='lower',
               cmap=plt.cm.viridis, title='', hold=False, **kwargs):
        
        m = numpy.isfinite(self)
        
        if vmin is None:                        
            vmin = numpy.percentile(self[m], 5.)
            
        if vmax is None:
            vmax = numpy.percentile(self[m], 95.)
            
        if ax is None:
            fig, ax = plt.subplots()

        map1 = ax.imshow(self, vmin=vmin, vmax=vmax, origin=origin, cmap=cmap, **kwargs)
        ax.set_title(title)

        if not hold:
            plt.show()

class DataLoader:
    '''
    DataLoader

    '''

    def __init__(self, science_img, header_ext=0):
        '''

        inputs
        -------
        science_img: str, path to the science image
        header_ext: int, extension number of the main header

        '''

        self.data = fits.open(science_img)
        self.data.info()

        self.num_ext = len(self.data)

    def read_ext(self, ext=1):
        '''

        inputs
        --------
        ext: int, extension number

        '''
        assert ext < self.num_ext, f'ext={ext} does not exist'
        return array(self.data[ext].data) # this will add method 'imshow'

    def get_keyword(self, keyword, ext=0):
        '''
        inputs
        --------
        keyword: str
        '''
        return self.data[ext].header.get(keyword)
