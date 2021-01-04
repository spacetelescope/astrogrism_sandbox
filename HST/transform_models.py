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
    Based on the JWST NIRISS transform model code, see
    https://github.com/spacetelescope/jwst/blob/master/jwst/transforms/models.py
    """

    standard_broadcasting = False
    _separable = False
    fittable = False
    linear = False

    n_inputs = 5
    n_outputs = 4

    def __init__(self, orders, lmodels=None, xmodels=None,
                 ymodels=None, theta=0., name=None, meta=None):
        self._order_mapping = {int(k): v for v, k in enumerate(orders)}
        self.xmodels = xmodels
        self.ymodels = ymodels
        self.lmodels = lmodels
        self.theta = theta
        self.orders = orders
        meta = {"orders": orders}
        if name is None:
            name = 'wfc3ir_forward_row_grism_dispersion'
        super(WFC3IRForwardGrismDispersion, self).__init__(name=name,
                                                              meta=meta)
        # starts with the backwards pixel and calculates the forward pixel
        self.inputs = ("x", "y", "x0", "y0", "order")
        self.outputs = ("x", "y", "wavelength", "order")

    def evaluate(self, x, y, x0, y0, order):
        """Return the valid pixel(s) and wavelengths given x0, y0, x, y, order
        Parameters
        ----------
        x0: int,float,list
            Source object x-center
        y0: int,float,list
            Source object y-center
        x :  int,float,list
            Input x location
        y :  int,float,list
            Input y location
        order : int
            Spectral order to use

        Returns
        -------
        x0, y0, lambda, order in the direct image for the pixel that was
        specified as input using the wavelength l and spectral order

        Notes
        -----
        I kept the possibility of having a rotation like NIRISS, although I
        don't know if there is a use case for it for WFC3.

        The two `flatten` lines may need to be uncommented if we want to use
        this for array input.
        """
        try:
            iorder = self._order_mapping[int(order.flatten()[0])]
        except AttributeError:
            iorder = self._order_mapping[order]
        except KeyError:
            raise ValueError("Specified order is not available")

        # The next two lines are to get around the fact that
        # modeling.standard_broadcasting=False does not work.
        #x00 = x0.flatten()[0]
        #y00 = y0.flatten()[0]

        t = np.linspace(0, 1, 10)  #sample t
        xmodel = self.xmodels[iorder]
        ymodel = self.ymodels[iorder]
        lmodel = self.lmodels[iorder]

        dx = xmodel.evaluate(x0, y0, t)
        dy = ymodel.evaluate(x0, y0, t)

        if self.theta != 0.0:
            rotate = Rotation2D(self.theta)
            dx, dy = rotate(dx, dy)

        so = np.argsort(dx)
        tab = Tabular1D(dx[so], t[so], bounds_error=False, fill_value=None)

        dxr = astmath.SubtractUfunc()
        wavelength = dxr | tab | lmodel
        model = Mapping((2, 3, 0, 2, 4)) | Const1D(x0) & Const1D(y0) & wavelength & Const1D(order)
        return model(x, y, x0, y0, order)



class WFC3IRBackwardGrismDispersion(Model):
    """Return the dispersed pixel(s) given center x, y, lambda, and order

    Parameters
    ----------
    xmodels : list[tuple]
        The list of tuple(models) for the polynomial model in x
    ymodels : list[tuple]
        The list of tuple(models) for the polynomial model in y
    lmodels : list
        The list of models for the polynomial model in l
    orders : list
        The list of orders which are available to the model
    theta : float
        Angle [deg] - defines the NIRISS filter wheel position

    Returns
    -------
    x, y, x0, y0, order in the grism image for the pixel at x0,y0 that was
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
                 ymodels=None, theta=None, name=None, meta=None):
        self._order_mapping = {int(k): v for v, k in enumerate(orders)}
        self.xmodels = xmodels
        self.ymodels = ymodels
        self.lmodels = lmodels
        self.orders = orders
        self.theta = theta
        meta = {"orders": orders}
        if name is None:
            name = 'wfc3ir_backward_grism_dispersion'
        super(WFC3IRBackwardGrismDispersion, self).__init__(name=name,
                                                            meta=meta)
        self.inputs = ("x", "y", "wavelength", "order")
        self.outputs = ("x", "y", "x0", "y0", "order")

    def evaluate(self, x, y, wavelength, order):
        """Return the dispersed pixel(s) given center x, y, lam and order
        Parameters
        ----------
        x :  int,float
            Input x location on the direct image
        y :  int,float
            Input y location on the direct image
        wavelength : float
            Wavelength to disperse
        order : list
            The order to use

        Returns
        -------
        x, y in the grism image for the pixel at x0, y0 that was
        specified as input using the wavelength and order specified

        Notes
        -----
        I kept the potential for rotation from NIRISS, unsure if it's actually
        needed/useful for WFC3. Original note:

        There's spatial dependence for NIRISS so the forward transform
        dependes on x,y as well as the filter wheel rotation. Theta is
        usu. taken to be the different between fwcpos_ref in the specwcs
        reference file and fwcpos from the input image.
        """
        if wavelength < 0:
            raise ValueError("Wavelength should be greater than zero")

        try:
            iorder = self._order_mapping[int(order.flatten()[0])]
        except AttributeError:
            iorder = self._order_mapping[order]
        except KeyError:
            raise ValueError("Specified order is not available")

        t = self.lmodels[iorder](wavelength)
        xmodel = self.xmodels[iorder]
        ymodel = self.ymodels[iorder]

        dx = xmodel.evaluate(x, y, t)
        dy = ymodel.evaluate(x, y, t)

        ## rotate by theta
        if self.theta != 0.0:
            rotate = Rotation2D(self.theta)
            dx, dy = rotate(dx, dy)

        return (x+dx, y+dy, x, y, order)
