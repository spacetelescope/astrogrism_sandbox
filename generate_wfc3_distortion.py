# The distortion reference file is much more complex than previous reference
# files we've had to convert. In an attempt to determine what information
# is required to gather from our Product Owner/external Subject Matter Experts,
# I'll attempt to break down the following code into its individual components,
# most derived from pysiaf

from asdf import AsdfFile
from astropy.modeling.models import Polynomial2D, Mapping, Shift
import astropy.units as u
from astropy.io import fits
from jwst.datamodels import DistortionModel#, util
from mirage.utils.siaf_interface import sci_subarray_corners
import numpy as np
import pysiaf

from stdatamodels import util

#import read_siaf_table

def get_distortion_coeffs(degree, filter_info):
    """Retrieve the requested set of distortion coefficients from Siaf
    and package into a dictionary
    Paramters
    ---------
    direction_label ; str
        Either 'Sci2Idl' or 'Idl2Sci'
    Returns
    -------
    x_coeffs : dict
        Dictionary containing x coefficients
    y_coeffs : dict
        Dictionary containing y coefficients
    """
    # Create dictionaries of distortion coefficients
    x_coeffs = {}
    y_coeffs = {}

    for i in range(1, degree+1):
        for j in range(0, i+1):
            xcolname = 'CX{}{}'.format(i, j)
            ycolname = xcolname.replace('X', 'Y')
            key = 'c{}_{}'.format(i-j, j)
            x_coeffs[key] = filter_info[xcolname]
            y_coeffs[key] = filter_info[ycolname]
    return x_coeffs, y_coeffs

def v2v3_model(from_sys, to_sys, par, angle):
    """
    Creates an astropy.modeling.Model object
    for the undistorted ("ideal") to V2V3 coordinate translation
    """
    if from_sys != 'v2v3' and to_sys != 'v2v3':
        raise ValueError("This function is designed to generate the transformation either to or from V2V3.")

    # Cast the transform functions as 1st order polynomials
    xc = {}
    yc = {}
    if to_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = np.sin(angle)
        yc['c1_0'] = (0.-par) * np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    if from_sys == 'v2v3':
        xc['c1_0'] = par * np.cos(angle)
        xc['c0_1'] = par * (0. - np.sin(angle))
        yc['c1_0'] = np.sin(angle)
        yc['c0_1'] = np.cos(angle)

    #0,0 coeff should never be used.
    xc['c0_0'] = 0
    yc['c0_0'] = 0

    xmodel = Polynomial2D(1, **xc)
    ymodel = Polynomial2D(1, **yc)

    return xmodel, ymodel

#https://github.com/spacetelescope/nircam_calib/blob/master/nircam_calib/reffile_creation/pipeline/distortion/nircam_distortion_reffiles_from_pysiaf.py#L37
def create_nircam_distortion(detector, outname, sci_pupil,
                             sci_subarr, sci_exptype, history_entry, filter):
    """
    Create an asdf reference file with all distortion components for the NIRCam imager.
    NOTE: The IDT has not provided any distortion information. The files are constructed
    using ISIM transformations provided/(computed?) by the TEL team which they use to
    create the SIAF file.
    These reference files should be replaced when/if the IDT provides us with distortion.
    Parameters
    ----------
    detector : str
        NRCB1, NRCB2, NRCB3, NRCB4, NRCB5, NRCA1, NRCA2, NRCA3, NRCA4, NRCA5
    aperture : str
        Name of the aperture/subarray. (e.g. FULL, SUB160, SUB320, SUB640, GRISM_F322W2)
    outname : str
        Name of output file.
    Examples
    --------
    """
    # Download WFC3 Image Distortion File
    from astropy.utils.data import download_file
    fn = download_file('https://hst-crds.stsci.edu/unchecked_get/references/hst/w3m18525i_idc.fits', cache=True)
    wfc3_distortion_file = fits.open(fn)
    wfc3_filter_info = wfc3_distortion_file[1].data[list(wfc3_distortion_file[1].data['FILTER']).index(filter)]
    
    
    degree = 4  # WFC3 Distortion is fourth degree
    
    # From Bryan Hilbert:
    #   The parity term is just an indicator of the relationship between the detector y axis and the “science” y axis.
    #   A parity of -1 means that the y axes of the two systems run in opposite directions... A value of 1 indicates no flip.
    # From Colin Cox:
    #   ... for WFC3 it is always -1 so maybe people gave up mentioning it.
    parity = -1
    
    #full_aperture = detector + '_' + aperture

    # Get Siaf instance for detector/aperture
    #inst_siaf = pysiaf.Siaf('nircam')
    #siaf = inst_siaf[full_aperture]

    # *****************************************************
    # "Forward' transformations. science --> ideal --> V2V3
    xcoeffs, ycoeffs = get_distortion_coeffs(degree, wfc3_filter_info)

    sci2idlx = Polynomial2D(degree, **xcoeffs)
    sci2idly = Polynomial2D(degree, **ycoeffs)

    # Get info for ideal -> v2v3 or v2v3 -> ideal model
    idl2v2v3x, idl2v2v3y = v2v3_model('ideal', 'v2v3', parity, np.radians(wfc3_distortion_file[1].data[wfc3_distortion_file[1].data['FILTER'] == filter]['THETA'][0]))

    '''
    # *****************************************************
    # 'Reverse' transformations. V2V3 --> ideal --> science
    xcoeffs, ycoeffs = get_distortion_coeffs('Idl2Sci', siaf)

    idl2scix = Polynomial2D(degree, **xcoeffs)
    idl2sciy = Polynomial2D(degree, **ycoeffs)

    # Get info for ideal -> v2v3 or v2v3 -> ideal model
    v2v32idlx, v2v32idly = v2v3_model('v2v3', 'ideal', parity, np.radians(wfc3_distortion_file['THETA']))
    '''

    # Now create a compound model for each with the appropriate inverse
    # Inverse polynomials were removed in favor of using GWCS' numerical inverse capabilities
    sci2idl = Mapping([0, 1, 0, 1]) | sci2idlx & sci2idly
    #sci2idl.inverse = Mapping([0, 1, 0, 1]) | idl2scix & idl2sciy

    idl2v2v3 = Mapping([0, 1, 0, 1]) | idl2v2v3x & idl2v2v3y
    #idl2v2v3.inverse = Mapping([0, 1, 0, 1]) | v2v32idlx & v2v32idly

    # Now string the models together to make a single transformation

    # We also need
    # to account for the difference of 1 between the SIAF
    # coordinate values (indexed to 1) and python (indexed to 0).
    # Nadia said that this shift should be present in the
    # distortion reference file.

    core_model = sci2idl# | idl2v2v3

    # Now add in the shifts to create the full model
    # including the shift to go from 0-indexed python coords to
    # 1-indexed

    # Find the distance between (0,0) and the reference location
    xshift = Shift(wfc3_filter_info['XREF'])
    yshift = Shift(wfc3_filter_info['YREF'])
    
    # Finally, we need to shift by the v2,v3 value of the reference
    # location in order to get to absolute v2,v3 coordinates
    v2shift = Shift(wfc3_filter_info['V2REF'])
    v3shift = Shift(wfc3_filter_info['V3REF'])
    
    # SIAF coords
    index_shift = Shift(1)
    model = index_shift & index_shift | xshift & yshift | core_model | v2shift & v3shift

    # Since the inverse of all model components are now defined,
    # the total model inverse is also defined automatically

    # Save using the DistortionModel datamodel
    d = DistortionModel(model=model, input_units=u.pix,
                        output_units=u.arcsec)

    #Populate metadata

    # Keyword values in science data to which this file should
    # be applied
    p_pupil = ''
    for p in sci_pupil:
        p_pupil = p_pupil + p + '|'

    p_subarr = ''
    for p in sci_subarr:
        p_subarr = p_subarr + p + '|'

    p_exptype = ''
    for p in sci_exptype:
        p_exptype = p_exptype + p + '|'

    d.meta.instrument.p_pupil = p_pupil
    d.meta.subarray.p_subarray = p_subarr
    d.meta.exposure.p_exptype = p_exptype

    # metadata describing the reference file itself
    d.meta.title = "WFC3 Distortion"
    d.meta.instrument.name = "WFC3"
    d.meta.instrument.module = detector[-2]
    
    numdet = detector[-1]
    d.meta.instrument.channel = "LONG" if numdet == '5' else "SHORT"
    # In the reference file headers, we need to switch NRCA5 to
    # NRCALONG, and same for module B.
    d.meta.instrument.detector = (detector[0:4] + 'LONG') if numdet == 5 else detector
    
    d.meta.telescope = 'HST'
    d.meta.subarray.name = 'FULL'
    d.meta.pedigree = 'GROUND'
    d.meta.reftype = 'DISTORTION'
    d.meta.author = 'D. Nguyen'
    d.meta.litref = "https://github.com/spacetelescope/jwreftools"
    d.meta.description = "Distortion model from SIAF coefficients in pysiaf version 0.6.1"
    #d.meta.exp_type = exp_type
    d.meta.useafter = "2014-10-01T00:00:00"

    # To be ready for the future where we will have filter-dependent solutions
    d.meta.instrument.filter = 'N/A'

    # Create initial HISTORY ENTRY
    sdict = {'name': 'nircam_distortion_reffiles_from_pysiaf.py',
             'author': 'B.Hilbert',
             'homepage': 'https://github.com/spacetelescope/jwreftools',
             'version': '0.8'}

    entry = util.create_history_entry(history_entry, software=sdict)
    d.history = [entry]

    #Create additional HISTORY entries
    #entry2 = util.create_history_entry(history_2)
    #d.history.append(entry2)

    d.save(outname)
    print("Output saved to {}".format(outname))


# Sample Invocation
# SW = "Short wavelength", lw = "Long Wavelength"

#https://github.com/spacetelescope/nircam_calib/blob/master/nircam_calib/reffile_creation/pipeline/distortion/make_all_imaging_distortion_reffiles_from_pysiaf.py#L49
import os

detector = 'WFC3IR'
#apname = 'FULL'
outname = '{}_distortion.asdf'.format(detector)# + '_' + apname)
pupil = ['NRC_IMAGE', 'NRC_TSIMAGE', 'NRC_FLAT', 'NRC_LED',
         'NRC_WFSC', 'NRC_TACQ', 'NRC_TACONFIRM', 'NRC_FOCUS',
         'NRC_DARK', 'NRC_WFSS', 'NRC_TSGRISM', 'NRC_GRISM']
subarr =['GENERIC']
exp_type = pupil
hist = "A Random Description"
filter = 'F105W'
#ref.create_nircam_distortion(detector, apname, outname, pupil, subarr, exp_type, hist)
create_nircam_distortion(detector, outname, pupil, subarr, exp_type, hist, filter)

