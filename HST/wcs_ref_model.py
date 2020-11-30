import warnings
from astropy import units as u
from stdatamodels.validate import ValidationWarning

from jwst.datamodels import ReferenceFileModel

class WFC3IRGrismModel(ReferenceFileModel):
    """
    A model for a reference file of type "specwcs" for HST IR grisms (G141 and
    G102). This reference file contains the models for wave, x, and y \
    polynomial solutions that describe dispersion through the grism.
    Parameters
    ----------
    displ: `~astropy.modeling.Model`
          HST Grism wavelength dispersion model
    dispx : `~astropy.modeling.Model`
          HST Grism row dispersion model
    dispy : `~astropy.modeling.Model`
          HST Grism column dispersion model
    invdispl : `~astropy.modeling.Model`
          HST Grism inverse wavelength dispersion model
    invdispx : `~astropy.modeling.Model`
          HST Grism inverse row dispersion model
    invdispy : `~astropy.modeling.Model`
          HST Grism inverse column dispersion model
    orders : `~astropy.modeling.Model`
          HST Grism orders, matched to the array locations of the
          dispersion models
    """
    schema_url = "http://stsci.edu/schemas/jwst_datamodel/specwcs_nircam_grism.schema"
    reftype = "specwcs"

    def __init__(self, init=None,
                       displ=None,
                       dispx=None,
                       dispy=None,
                       invdispl=None,
                       invdispx=None,
                       invdispy=None,
                       orders=None,
                       **kwargs):
        super(WFC3IRGrismModel, self).__init__(init=init, **kwargs)

        if init is None:
            self.populate_meta()
        if displ is not None:
            self.displ = displ
        if dispx is not None:
            self.dispx = dispx
        if dispy is not None:
            self.dispy = dispy
        if invdispl is not None:
            self.invdispl = invdispl
        if invdispx is not None:
            self.invdispx = invdispx
        if invdispy is not None:
            self.invdispy = invdispy
        if orders is not None:
            self.orders = orders

    def populate_meta(self):
        self.meta.instrument.name = "WFC3"
        self.meta.exposure.type = "IR_GRISM"
        self.meta.reftype = self.reftype

    def validate(self):
        super(WFC3IRGrismModel, self).validate()
        try:
            assert isinstance(self.meta.input_units, (str, u.NamedUnit))
            assert isinstance(self.meta.output_units, (str, u.NamedUnit))
            assert self.meta.instrument.name == "WFC3"
            assert self.meta.exposure.type == "IR_GRISM"
            assert self.meta.reftype == self.reftype
        except AssertionError as errmsg:
            if self._strict_validation:
                raise AssertionError(errmsg)
            else:
                warnings.warn(str(errmsg), ValidationWarning)

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")
