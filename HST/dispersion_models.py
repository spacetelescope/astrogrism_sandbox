import numpy as np
from astropy.modeling.models import custom_model
from astropy.modeling import Model, Parameter
from asdf.extension import Converter

class DISPXY_Model(Model):
    n_inputs = 3
    n_outputs = 1
    _tag = "tag:stsci.edu:grismstuff/dispxy_model-1.0.0"
    _name = "DISPXY_Model"

    def __init__(self, ematrix, offset, inv=False):
        self.ematrix = np.array(ematrix)
        self.inv = inv
        self.offset = offset

    # Note that in the inverse case, input "t" here is actually dx or dy
    def evaluate(self, x, y, t):
        e = self.ematrix
        offset = self.offset
        coeffs = np.array([1, x, y, x**2, x*y, y**2])
        if self.inv:
            return (t + offset - np.dot(coeffs, e[0,:]))/np.dot(coeffs, e[1,:])
        else:
            return np.dot(coeffs, e[0,:]) + t*np.dot(coeffs, e[1,:]) - offset

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
