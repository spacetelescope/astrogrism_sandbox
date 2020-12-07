import math
from collections import namedtuple
import numpy as np
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter, InputParameterError
from astropy.modeling.models import (Rotation2D, Identity, Mapping, Tabular1D, Const1D)
from astropy.modeling.models import math as astmath
from astropy.utils import isiterable


class WFC3IRForwardGrismDispersion(Model):
    """Return the transform from grism to image for the given spectral order.

    Parameters
    ----------
    orders : list [int]
        List of orders which are available

    lmodels : list [astropy.modeling.Model]
        List of models which govern the wavelength solutions for each order

    xmodels : list [astropy.modeling.Model]
        List of models which govern the x solutions for each order

    ymodels : list [astropy.modeling.Model]
        List of models which givern the y solutions for each order

    Returns
    -------
    x, y, wavelength, order in the grism image for the pixel at x0,y0 that was
    specified as input using the input delta pix for the specified order

    Notes
    -----
    Based on the JWST transform model code, see
    https://github.com/spacetelescope/jwst/blob/master/jwst/transforms/models.py
    """
    standard_broadcasting = False
    _separable = False
    fittable = False
    linear = False

    n_inputs = 5
    n_outputs = 4

    def __init__(self, orders, lmodels=None, xmodels=None,
                 ymodels=None, name=None, meta=None):
        self.orders = orders
        self.lmodels = lmodels
        self.xmodels = xmodels
        self.ymodels = ymodels
        self._order_mapping = {int(k): v for v, k in enumerate(orders)}
        meta = {"orders": orders}  # informational for users
        if name is None:
            name = 'wfc3ir_forward_grism_dispersion'
        super(WFC3IRForwardGrismDispersion, self).__init__(name=name,
                                                              meta=meta)
        self.inputs = ("x", "y", "x0", "y0", "order")
        self.outputs = ("x", "y", "wavelength", "order")

    def evaluate(self, x, y, x0, y0, order):
        """Return the transform from grism to image for the given spectral order.

        Parameters
        ----------
        x : float
            input x pixel
        y : float
            intput y pixel
        x0 : float
            input x-center of object
        y0 : float
            input y-center of object
        order : int
            the spectral order to use
        """
        try:
            iorder = self._order_mapping[int(order.flatten()[0])]
        except KeyError:
            raise ValueError("Specified order is not available")

        xmodel = self.xmodels[iorder]
        ymodel = self.ymodels[iorder]
        lmodel = self.lmodels[iorder]

        # inputs are x, y, x0, y0, order

        tmodel = astmath.SubtractUfunc() | xmodel
        model = Mapping((0, 2, 0, 2, 2, 3, 4)) | ( tmodel | ymodel) & (tmodel | lmodel) & Identity(3) |\
              Mapping((2, 3, 0, 1, 4)) | Identity(1) & astmath.AddUfunc() &  Identity(2) | Mapping((0, 1, 2, 3), n_inputs=4)

        return model(x, y, x0, y0, order)

class WFC3IRBackwardGrismDispersion(Model):
    """Return the valid pixel(s) and wavelengths given center x,y and lam

    Parameters
    ----------
    orders : list [int]
        List of orders which are available

    lmodels : list [astropy.modeling.Model]
        List of models which govern the wavelength solutions

    xmodels : list [astropy.modeling.Model]
        List of models which govern the x solutions

    ymodels : list [astropy.modeling.Model]
        List of models which givern the y solutions

    Returns
    -------
    x, y, lam, order in the grism image for the pixel at x0,y0 that was
    specified as input using the wavelength l for the specified order

    Notes
    -----
    Based on the JWST transform model code, see
    https://github.com/spacetelescope/jwst/blob/master/jwst/transforms/models.py
    """
    standard_broadcasting = False
    _separable = False
    fittable = False
    linear = False

    n_inputs = 4
    n_outputs = 5

    def __init__(self, orders, lmodels=None, xmodels=None,
                 ymodels=None, name=None, meta=None):
        self._order_mapping = {int(k): v for v, k in enumerate(orders)}
        self.lmodels = lmodels
        self.xmodels = xmodels
        self.ymodels = ymodels
        self.orders = orders
        meta = {"orders": orders}
        if name is None:
            name = "wfc3ir_backward_grism_dispersion"
        super(WFC3IRBackwardGrismDispersion, self).__init__(name=name,
                                                            meta=meta)
        self.inputs = ("x", "y", "wavelength", "order")
        self.outputs = ("x", "y", "x0", "y0", "order")

    def evaluate(self, x, y, wavelength, order):
        """Return the transform from image to grism for the given spectral order.

        Parameters
        ----------
        x : float
            input x pixel on the direct image
        y : float
            intput y pixel on the direct image
        wavelength : float
            input wavelength in angstroms
        order : int
            specifies the spectral order
        """
        try:
            iorder = self._order_mapping[int(order.flatten()[0])]
        except KeyError:
            raise ValueError("Specified order is not available")

        if (wavelength < 0).any():
            raise ValueError("wavelength should be greater than zero")

        # These should be dispx and dispy from the reference file
        xmodel = self.xmodels[iorder]
        ymodel = self.ymodels[iorder]
        # lmodel should be invdispl from the reference file
        lmodel = self.lmodels[iorder]

        # Convert lambda to t, then use that to get dx and dy to add to the
        # input coordinates to get the dispersed coordinates
        model = (Identity(2) * lmodel * Identity(1)) | \
                Mapping((0, 1, 2, 0, 1, 2, 0, 1, 3)) | \
                xmodel & ymodel & Identity(3) | \
                Mapping((0, 2, 1, 3, 2, 3, 4)) | \
                astmath.AddUfunc() & astmath.AddUfunc() & Identity(3)

        return model(x, y, wavelength, order)
