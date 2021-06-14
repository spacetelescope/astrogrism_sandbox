## Base class for Astrogrism
from importlib.resources import path as resource_path

import asdf
from astropy.io import fits
from astropy.modeling import models
from astropy import units as u
from gwcs import wcs as gwcs
from gwcs import coordinate_frames as cf
import numpy as np

from HST.transform_models import WFC3IRForwardGrismDispersion, WFC3IRBackwardGrismDispersion
from HST import DISPXY_Extension

# Will remove this hardcoded path once I have this packaged up for use with importlib
pkg_dir = '/Users/rosteen/projects/astrogrism_sandbox'

class GrismObs():
    """
    Base class for astrogrism package. Stores all necessary information about
    a single grism observation.
    """

    def __init__(self, grism_image, direct_image=None, telescope=None, instrument=None,
                 detector=None, filter=None):

        # Read grism image file if string input
        if isinstance(grism_image, str):
            self.grism_image = fits.open(grism_image)
        elif isinstance(grism_image, fits.HDUList):
            self.grism_image = grism_image
        else:
            raise TypeError("grism_image must be either a string filepath or FITS HDUList")

        # Read direct image file if string input
        if direct_image is None:
            self.direct_image = None
        if isinstance(direct_image, str):
            self.direct_image = fits.open(direct_image)
        elif isinstance(direct_image, fits.HDUList) or direct_image is None:
            self.direct_image = direct_image
        else:
            raise TypeError("direct_image must be either a string filepath or FITS HDUList")

        # Parse grism image file header for meta info
        self.grism_header = self.grism_image["PRIMARY"].header

        # Attempt to retrieve any information missing from the header (e.g. SIP)
        # Should probably make these properties instead.
        if telescope is None:
            self.telescope = self.grism_header["TELESCOP"]
        else:
            self.telescope = telescope

        if instrument is None:
            self.instrument = self.grism_header["INSTRUME"]
        else:
            self.instrument = instrument

        if filter is None:
            self.filter = self.grism_header["FILTER"]
        else:
            self.filter = filter


        # Build GWCS geometric transform pipeline
        self._build_geometric_transforms()

    def _build_geometric_transforms(self):

        """
        Build transform pipeline under the hood so the user doesn't need
        to worry about it.

        TODO:
        - Try to get SIP coefficients from grism observation header before
        resorting to premade file
        """

        # Register custom asdf extension
        asdf.get_config().add_extension(DISPXY_Extension())

        # Get paths to premade configuration files
        config_dir = "{}/config/{}/".format(pkg_dir, self.telescope)

        if self.telescope == "HST":
            if self.filter in ("G102", "G141"):
                instrument = self.instrument + "_IR"
            elif self.filter == "G280":
                instrument = self.instrument + "_UV"
        else:
            instrument = self.instrument
        sip_file = "{}/{}_distortion.fits".format(config_dir,
                                                  instrument)
        spec_wcs_file = "{}/{}_{}_specwcs.asdf".format(config_dir,
                                                       self.instrument,
                                                       self.filter)

        # Build the grism_detector <-> detector transforms
        specwcs = asdf.open(spec_wcs_file).tree
        displ = specwcs['displ']
        dispx = specwcs['dispx']
        dispy = specwcs['dispy']
        invdispl = specwcs['invdispl']
        invdispx = specwcs['invdispx']
        invdispy = specwcs['invdispy']
        orders = specwcs['order']

        gdetector = cf.Frame2D(name='grism_detector',
                       axes_order=(0, 1),
                       unit=(u.pix, u.pix))
        det2det = WFC3IRForwardGrismDispersion(orders,
                                               lmodels=displ,
                                               xmodels=invdispx,
                                               ymodels=dispy)
        det2det.inverse = WFC3IRBackwardGrismDispersion(orders,
                                                        lmodels=invdispl,
                                                        xmodels=dispx,
                                                        ymodels=dispy)

        grism_pipeline = [(gdetector, det2det)]

        # Now add the detector -> world transform
        sip_hdus = fits.open(sip_file)

        acoef = dict(sip_hdus[1].header['A_*'])
        a_order = acoef.pop('A_ORDER')
        bcoef = dict(sip_hdus[1].header['B_*'])
        b_order = bcoef.pop('B_ORDER')

        # Get the inverse SIP polynomial coefficients from file
        apcoef = dict(sip_hdus[1].header['AP_*'])
        bpcoef = dict(sip_hdus[1].header['BP_*'])

        try:
            ap_order = apcoef.pop('AP_ORDER')
            bp_order = bpcoef.pop('BP_ORDER')
        except ValueError:
            raise

        crpix = [sip_hdus[1].header['CRPIX1'], sip_hdus[1].header['CRPIX2']]

        crval = [self.grism_image[1].header['CRVAL1'],
                 self.grism_image[1].header['CRVAL2']]

        cdmat = np.array([[sip_hdus[1].header['CD1_1'], sip_hdus[1].header['CD1_2']],
                  [sip_hdus[1].header['CD2_1'], sip_hdus[1].header['CD2_2']]])

        a_polycoef = {}
        for key in acoef:
            a_polycoef['c' + key.split('A_')[1]] = acoef[key]

        b_polycoef = {}
        for key in bcoef:
            b_polycoef['c' + key.split('B_')[1]] = bcoef[key]

        ap_polycoef = {}
        for key in apcoef:
            ap_polycoef['c' + key.split('AP_')[1]] = apcoef[key]

        bp_polycoef = {}
        for key in bpcoef:
             bp_polycoef['c' + key.split('BP_')[1]] = bpcoef[key]

        a_poly = models.Polynomial2D(a_order, **a_polycoef)
        b_poly = models.Polynomial2D(b_order, **b_polycoef)
        ap_poly = models.Polynomial2D(ap_order, **ap_polycoef)
        bp_poly = models.Polynomial2D(bp_order, **bp_polycoef)

        # See SIP definition paper for definition of u, v, f, g
        SIP_forward = (models.Shift(-(crpix[0]-1)) & models.Shift(-(crpix[1]-1)) | # Calculate u and v
             models.Mapping((0, 1, 0, 1, 0, 1)) | a_poly & b_poly & models.Identity(2) |
             models.Mapping((0, 2, 1, 3)) | models.math.AddUfunc() & models.math.AddUfunc() |
             models.AffineTransformation2D(matrix=cdmat) | models.Pix2Sky_TAN() |
             models.RotateNative2Celestial(crval[0], crval[1], 180))

        SIP_backward = (models.RotateCelestial2Native(crval[0], crval[1], 180) |
            models.Sky2Pix_TAN() | models.AffineTransformation2D(matrix=cdmat).inverse |
            models.Mapping((0, 1, 0, 1, 0, 1)) | ap_poly & bp_poly & models.Identity(2) |
            models.Mapping((0, 2, 1, 3)) | models.math.AddUfunc() & models.math.AddUfunc() |
            models.Shift((crpix[0]-1)) & models.Shift((crpix[1]-1)))

        full_distortion_model = SIP_forward & models.Identity(2)
        full_distortion_model.inverse = SIP_backward & models.Identity(2)

        imagepipe = []

        det_frame = cf.Frame2D(name="detector")
        imagepipe.append((det_frame, full_distortion_model))

        world_frame = cf.CelestialFrame(name="world", unit = (u.Unit("deg"), u.Unit("deg")),
                             axes_names=('lon', 'lat'), axes_order=(0, 1),
                             reference_frame="ICRS")
        imagepipe.append((world_frame, None))

        grism_pipeline.extend(imagepipe)

        self.geometric_transforms = gwcs.WCS(grism_pipeline)
