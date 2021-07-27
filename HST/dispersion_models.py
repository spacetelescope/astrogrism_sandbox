import numpy as np
from scipy.interpolate import interp1d
from astropy.modeling.models import custom_model, Tabular1D
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

        if self.ematrix.shape == (2,):
            # Reshape to add second dimension for ematrix with only two values
            self.ematrix = np.reshape(self.ematrix, [2,1])

        if len(self.ematrix.shape) > 1:
            if self.inv and self.ematrix.shape[1] > 2:
                # Can't invert these here, need to interpolate from the other direction
                raise ValueError("Can't invert higher order coefficient matrices")

    # Note that in the inverse case, input "t" here is actually dx or dy
    def evaluate(self, x, y, t):
        inv = self.inv

        e = self.ematrix
        offset = self.offset
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                x = float(x)
            else:
                raise ValueError(f"x is array: {x}")
        if isinstance(y, np.ndarray):
            if len(y) == 1:
                y = float(y)
            else:
                raise ValueError(f"y is array: {y}")

        coeffs = {1: np.array([1]),
                  6: np.array([1, x, y, x**2, x*y, y**2])}

        t_order = e.shape[0]
        if len(e.shape) == 1:
            c_order = 1
        else:
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
