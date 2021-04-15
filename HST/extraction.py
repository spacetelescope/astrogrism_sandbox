import ndcube
import numpy as np
from astropy.io import fits
import asdf

from dispersion_models import DISPXY_Model, DISPXY_Extension
from transform_models import WFC3IRForwardGrismDispersion, WFC3IRBackwardGrismDispersion

def extract_2d_spectrum(data, ll_x, ll_y, ur_x, ur_y, ll_l = 1.0,
                      ur_l = 3.0, order = 1, specwcs_ref = "wfc3_ir_specwcs.asdf"):
    """
    Function to do a simple box cutout around a 2D spectrum.

    The input lower and upper x and y bounds are pixel coordinates from the
    direct image. The upper and lower wavelength bounds are in microns.
    """
    asdf.get_config().add_extension(DISPXY_Extension())
    specwcs = asdf.open(specwcs_ref).tree
    displ = specwcs['displ']
    dispx = specwcs['dispx']
    dispy = specwcs['dispy']
    invdispl = specwcs['invdispl']
    invdispx = specwcs['invdispx']
    invdispy = specwcs['invdispy']
    orders = specwcs['order']

    det2det = WFC3IRForwardGrismDispersion(orders,
                                           lmodels=displ,
                                           xmodels=invdispx,
                                           ymodels=dispy)
    det2det.inverse = WFC3IRBackwardGrismDispersion(orders,
                                                    lmodels=invdispl,
                                                    xmodels=dispx,
                                                    ymodels=dispy)

    ll = det2det.inverse.evaluate(ll_x, ll_y, ll_l, order)
    ur = det2det.inverse.evaluate(ur_x, ur_y, ur_l, order)

    print(ll, ur)

    return data[int(ll[1]):int(ur[1])+1, int(ll[0]):int(ur[0])+1]

