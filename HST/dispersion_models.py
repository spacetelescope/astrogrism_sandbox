import numpy as np
from scipy.interpolate import interp1d
from astropy.modeling.models import custom_model
from astropy.modeling import Model, Parameter
from asdf.extension import Converter


class interp1d_picklable(object):
    """ class wrapper for piecewise linear function
    """

    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interp1d(state[0], state[1], **state[2])


class DISPXY_Model(Model):
    n_inputs = 3
    n_outputs = 1
    _tag = "tag:stsci.edu:grismstuff/dispxy_model-1.0.0"
    _name = "DISPXY_Model"

    def __init__(self, ematrix, offset, inv=False, interpolate=False):
        self.ematrix = np.array(ematrix)
        self.inv = inv
        self.interpolate = interpolate
        self.offset = offset

        if len(self.ematrix.shape) > 1:
            if self.inv and self.ematrix.shape[1] > 2:
                # Have to do the inverse transform for higher order in t via interpolation
                self.interpolate=True

    # Note that in the inverse case, input "t" here is actually dx or dy
    # t0 is only used in the case of interpolating for an inverse transform
    def evaluate(self, x, y, t, t0=np.linspace(-1,2,40), inv=None, interpolate=None):
        # Need this to allow override of inverse and interpolate for recursion
        if inv is None:
            inv = self.inv
        if interpolate is None:
            interpolate = self.interpolate

        e = self.ematrix
        offset = self.offset
        coeffs = {1: np.array([1]),
                  6: np.array([1, x, y, x**2, x*y, y**2])}

        t_order = e.shape[0]
        c_order = e.shape[1]

        f = 0

        if inv:
            if interpolate:
                xr, yr = self.evaluate(x, y, t0, inv=False)
                so = np.argsort(yr)
                interpolation = interp1d_picklable(yr[so], t0[so])
                f = interpolation(t)
            else:
                f = ((t + offset - np.dot(coeffs[c_order], e[0,:])) /
                     np.dot(coeffs[c_order], e[1,:]))
        else:
            for i in range(0, t_order):
                f += t**i * (np.dot(coeffs[c_order], e[i,:]))

        return f

class DISPXY_ModelConverter(Converter):
    tags = ["tag:stsci.edu:grismstuff/dispxy_model-*"]
    types = [DISPXY_Model]

    def to_yaml_tree(self, obj, tags, ctx):
        # ASDF will know how to turn the nested lists into yaml properly
        return {"ematrix": obj.ematrix, "inverse_flag": obj.inv}

    def from_yaml_tree(self, node, tags, ctx):
        ematrix = node['ematrix']
        inverse_flag = node['inverse_flag']
        return DISPXY_Model(ematrix, inverse_flag)

class DISPXY_Extension():
    extension_uri = "asdf://stsci.edu/grismstuff/extensions/extension-1.0"
    converters = [DISPXY_ModelConverter()]
    tags = ["tag:stsci.edu:grismstuff/dispxy_model-1.0.0"]
    #tags = ["asdf://stsci.edu/grismstuff/tags/dispxy_model-1.0"]
