import numpy as np
from astropy.modeling.models import custom_model

def create_dispxy_model(e, inverse=False):
    """Return a custom astropy model defining the dispersion.

    Equations here are specifically for WFC3 IR, taken from GRISMCONF's poly.py
    file (specifically POLY12 and INVPOLY12). Note that the equations are of
    the same form for both x and y, differing only in the input coefficient
    matrix e.
    """
    e = np.array(e)
    print(e)

    @custom_model
    def disp_model(x, y, t):
        coeffs = [1, x, y, x**2, x*y, y**2]
        if inverse:
            return (t - np.dot(coeffs,e[0,:]))/np.dot(coeffs,e[1,:])
        else:
            return np.dot(coeffs,e[0,:]) + t*np.dot(coeffs,e[1,:])

    return disp_model
