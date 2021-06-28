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

        if len(self.ematrix.shape) > 1:
            if self.inv and self.ematrix.shape[1] > 2:
                print(self.ematrix.shape)
                raise NotImplementedError("Cannot create inverse dispersion transform"
                                          " for higher order than linear in t")

    # Note that in the inverse case, input "t" here is actually dx or dy
    def evaluate(self, x, y, t):
        e = self.ematrix
        offset = self.offset
        coeffs = {1: np.array([1]),
                  6: np.array([1, x, y, x**2, x*y, y**2])}

        t_order = e.shape[0]
        c_order = e.shape[1]

        f = 0

        if self.inv:
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
