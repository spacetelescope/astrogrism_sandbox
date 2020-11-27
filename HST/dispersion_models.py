import numpy as np
from astropy.modeling.models import custom_model
from astropy.modeling import Model, Parameter
from asdf.extension import Converter

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

class DISPXY_Model(Model):
    n_inputs = 3
    n_outputs = 1

    def __init__(self, ematrix, inv=False):
        self.ematrix = np.array(ematrix)
        self.inv = inv

    def evaluate(self, x, y, t):
        e = self.ematrix
        coeffs = np.array([1, x, y, x**2, x*y, y**2])
        if self.inv:
            return (t - np.dot(coeffs, e[0,:]))/np.dot(coeffs, e[1,:])
        else:
            return np.dot(coeffs, e[0,:]) + t*np.dot(coeffs, e[1,:])

class DISPXY_ModelConverter(Converter):
    tags = ["tag:stsci.edu:grismstuff/dispxy_model-*"]
    types = ["mypackage.DispxyFunction"]

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
    tags = ["asdf://stsci.edu/grismstuff/tags/dispxy_model-1.0"]
