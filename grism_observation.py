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
        crpix = [sip_hdus[1].header['CRPIX1'], sip_hdus[1].header['CRPIX2']]

        crval = [self.grism_image[1].header['CRVAL1'],
                 self.grism_image[1].header['CRVAL2']]

        cdmat = np.array([[sip_hdus[1].header['CD1_1'], sip_hdus[1].header['CD1_2']],
                  [sip_hdus[1].header['CD2_1'], sip_hdus[1].header['CD2_2']]])

        apcoef = {}
        for key in acoef:
            apcoef['c' + key.split('A_')[1]] = acoef[key]

        bpcoef = {}
        for key in bcoef:
            bpcoef['c' + key.split('B_')[1]] = bcoef[key]

        a_poly = models.Polynomial2D(a_order, **apcoef)
        b_poly = models.Polynomial2D(b_order, **bpcoef)

        # See SIP definition paper for definition of u, v, f, g
        mr = (models.Shift(-(crpix[0]-1)) & models.Shift(-(crpix[1]-1)) | # Calculate u and v
             models.Mapping((0, 1, 0, 1, 0, 1)) | a_poly & b_poly & models.Identity(2) |
             models.Mapping((0, 2, 1, 3)) | models.math.AddUfunc() & models.math.AddUfunc() |
             models.AffineTransformation2D(matrix=cdmat) | models.Pix2Sky_TAN() |
             models.RotateNative2Celestial(crval[0], crval[1], 180))

        imagepipe = []

        det_frame = cf.Frame2D(name="detector")
        imagepipe.append((det_frame, mr & models.Identity(2)))

        world_frame = cf.CelestialFrame(name="world", unit = (u.Unit("deg"), u.Unit("deg")),
                             axes_names=('lon', 'lat'), axes_order=(0, 1),
                             reference_frame="ICRS")
        imagepipe.append((world_frame, None))

        grism_pipeline.extend(imagepipe)

        self.geometric_transforms = gwcs.WCS(grism_pipeline)
