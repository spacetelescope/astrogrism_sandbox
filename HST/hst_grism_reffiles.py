import re
import datetime
import numpy as np

import asdf
from asdf.tags.core import Software, HistoryEntry

from astropy import units as u
from astropy.modeling.models import Polynomial1D

#from jwst.datamodels import NIRCAMGrismModel
from wcs_ref_model import WFC3IRGrismModel
from jwst.datamodels import wcs_ref_models
from dispersion_models import DISPXY_Model, DISPXY_Extension

def common_reference_file_keywords(reftype=None,
                                   title=None,
                                   description=None,
                                   exp_type=None,
                                   author="STScI",
                                   useafter="2014-01-01T00:00:00",
                                   fname=None,
                                   pupil=None, **kwargs):
    """
    exp_type can be also "N/A", or "ANY".
    """
    if exp_type is None:
        raise ValueError("exp_type not set")
    if reftype is None:
        raise ValueError("Expected reftype value")

    ref_file_common_keywords = {
        "author": author,
        "description": description,
        "exposure": {"type": exp_type},
        "instrument": {"name": "WFC3"},
        "pedigree": "ground",
        "reftype": reftype,
        "telescope": "HST",
        "title": title,
        "useafter": useafter,
        }

    if fname is not None:
        ref_file_common_keywords["instrument"]["filter"] = fname
    if pupil is not None:
        ref_file_common_keywords["instrument"]["pupil"] = pupil

    ref_file_common_keywords.update(kwargs)
    return ref_file_common_keywords


def create_grism_specwcs(conffile="",
                         pupil=None,
                         direct_filter=None,
                         author="STScI",
                         history="",
                         outname=None):
    """
    Note: This code is shamelessly stolen from the jwreftools package
    (see https://github.com/spacetelescope/jwreftools/) and adapted for use
    on HST GRISMCONF files. The docstrings and comments have not yet been
    updated accordingly.

    Create an asdf reference file to hold grism configuration information. No
    sensitivity information is included

    Note: The orders are named alphabetically, i.e. Order A, Order B
    There are also sensativity fits files which are tables of wavelength,
    sensativity, and error. These are specified in the conffile but will
    not be read in and saved in the output reference file for now.
    It's possible they may be included in the future, either here or as
    a separate reference files. Their use here would be to help define the
    min and max wavelengths which set the extent of the dispersed trace on
    the grism image. Convolving the sensitiviy file with the filter throughput
    allows one to calculate the wavelength of minimum throughput which defines
    the edges of the trace.

    direct_filter is not specified because it assumes that the wedge
    information (wx,wy) is included in the conf file in one of the key-value
    pairs, where the key includes the beam designation

     this reference file also contains the polynomial model which is
     appropriate for the coefficients which are listed.
     wavelength = DISPL(order,x0,y0,t)
     dx = DISPX(order,x0,y0,t)
     dy = DISPY(order,x0,y0,t)

     t = INVDISPX(order,x0,y0,dx)
     t = INVDISPY(order,x0,y0,dy)
     t = INVDISL(order,x0,y0, wavelength)



    Parameters
    ----------
    conffile : str
        The text file with configuration information, formatted as aXe expects
    pupil : str
        Name of the grism the conffile corresponds to
        Taken from the conffile name if not specified
    module : str
        Name of the Nircam module
        Taken from the conffile name if not specified
    author : str
        The name of the author
    history : str
        A comment about the refrence file to be saved with the meta information
    outname : str
        Output name for the reference file

    Returns
    -------
    fasdf : asdf.AsdfFile(WFC3IRGrismModel)

    """
    if outname is None:
        outname = "wfc3_ir_specwcs.asdf"
    if not history:
        history = "Created from {0:s}".format(conffile)

    # if pupil is none get from filename like NIRCAM_modB_R.conf
    if pupil is None:
        pupil = "GRISM" + conffile.split(".")[0][-1]
    print("Pupil is {}".format(pupil))

    ref_kw = common_reference_file_keywords(reftype="specwcs",
                                            title="HST IR Grism Parameters",
                                            description="{0:s} dispersion models".format(pupil),
                                            exp_type="WFC3_IR",
                                            author=author,
                                            model_type="WFC3IRGrismModel",
                                            fname=direct_filter,
                                            pupil=pupil,
                                            filename=outname,
                                            )

    # get all the key-value pairs from the input file
    conf = dict_from_file(conffile)
    beamdict = split_order_info(conf)

    # Get x and y offsets from the filter, if necessary
    if direct_filter is not None:
        wx = beamdict["WEDGE"][direct_filter][0]
        wy = beamdict["WEDGE"][direct_filter][0]
    else:
        wx, wy = 0, 0

    beamdict.pop("WEDGE")

    # beam = re.compile('^(?:[+\-]){0,1}[a-zA-Z0-9]{0,1}$')  # match beam only
    # read in the sensitivity tables to save their content
    # they currently have names like this: NIRCam.A.1st.sensitivity.fits
    # translated as inst.beam/order.param
    temp = dict()
    etoken = re.compile("^[a-zA-Z]*_(?:[+\-]){1,1}[1,2]{1,1}")  # find beam key
    for b, bdict in beamdict.items():
            temp[b] = dict()

    # add the new beam information to beamdict and remove spurious beam info
    for k in temp:
        for kk in temp[k]:
            if etoken.match(kk):
                kk = kk.replace("_{}".format(k), "")
            beamdict[k][kk] = temp[k][kk]

    # for NIRCAM, the R and C grism coefficients contain zeros where
    # the dispersion is in the opposite direction. Meaning, the GRISMR,
    # which disperses along ROWS has coefficients of zero in the y models
    # and vice versa.
    #
    # There are separate reference files for each grism. Depending on the grism
    # dispersion direction you either want to use the dx from source center or
    # the dy from source center in the inverse dispersion relationship which is
    # used to calculate the t value needed to calculate the wavelength at that
    # pixel.
    # The model creation here takes all of this into account by looking at the
    # GRISM[R/C] the file is used for and creating a reference model with the
    # appropriate dispersion direction in use. This eliminates having to decide
    # which direction to calculatethe dispersion from given the input x,y
    # pixel in the dispersed image.
    orders = beamdict.keys()

    # dispersion models valid per order and direction saved to reference file
    # Forward
    invdispl = []
    invdispx = []
    invdispy = []
    # Backward
    displ = []
    dispx = []
    dispy = []

    for order in orders:
        # convert the displ wavelengths to microns if the input
        # file is still in angstroms
        l0 = beamdict[order]['DISPL'][0] / 10000.
        l1 = beamdict[order]['DISPL'][1] / 10000.

        # create polynomials using the coefficients of each order

        # This holds the wavelength lookup coeffs
        # This model is  INVDISPL for backward and returns t
        # This model should be DISPL for forward and returns wavelength
        if l1 == 0:
            lmodel = Polynomial1D(1, c0=0, c1=0)
        else:
            lmodel = Polynomial1D(1, c0=-l0/l1, c1=1./l1)
        invdispl.append(lmodel)
        lmodel = Polynomial1D(1, c0=l0, c1=l1)
        displ.append(lmodel)

        # This holds the x coefficients, for the R grism this model is the
        # the INVDISPX returning t, for the C grism this model is the DISPX
        e = beamdict[order]['DISPX']
        xmodel = DISPXY_Model(e, wx)
        dispx.append(xmodel)
        inv_xmodel = DISPXY_Model(e, wx, inv=True)
        invdispx.append(inv_xmodel)

        # This holds the y coefficients, for the C grism, this model is
        # the INVDISPY, returning t, for the R grism, this model is the DISPY
        e = beamdict[order]['DISPY']
        ymodel = DISPXY_Model(e, wy)
        dispy.append(ymodel)
        inv_ymodel = DISPXY_Model(e, wy, inv=True)
        invdispy.append(ymodel)

    # change the orders into translatable integers
    # so that we can look up the order with the proper index
    oo = [int(o) for o in beamdict]

    # We need to register the converter for the DISPXY_Model class with asdf
    asdf.get_config().add_extension(DISPXY_Extension())

    ref = WFC3IRGrismModel()
    ref.meta.update(ref_kw)
    # This reference file is good for NRC_WFSS and TSGRISM modes
    ref.meta.exposure.p_exptype = "NRC_WFSS|NRC_TSGRISM"
    ref.meta.input_units = u.micron
    ref.meta.output_units = u.micron
    ref.displ = displ
    ref.dispx = dispx
    print(ref.dispx)
    ref.dispy = dispy
    print(ref.dispy)
    ref.invdispx = invdispx
    ref.invdispy = invdispy
    ref.invdispl = invdispl
    ref.order = oo
    history = HistoryEntry({'description': history,
                            'time': datetime.datetime.utcnow()})
    software = Software({'name': 'nircam_reftools.py',
                         'author': author,
                         'homepage': 'https://github.com/spacetelescope/jwreftools',
                         'version': '0.7.1'})
    history['software'] = software
    ref.history = [history]
    ref.to_asdf(outname)
    ref.validate()


def create_tsgrism_wavelengthrange(outname="wfc3_tsgrism_wavelengthrange.asdf",
                                   history="WFC3 TSGrism wavelengthrange",
                                   author="STScI",
                                   wavelengthrange=None,
                                   extract_orders=None):
    """Create a wavelengthrange reference file for WFC3 TSGRISM mode.

    Parameters
    ----------
    outname: str
        The output name of the file
    history: str
        History information about it's creation
    author: str
        Person or entity making the file
    wavelengthrange: list(tuples)
        A list of tuples that set the order, filter, and
        wavelength range min and max
    extract_orders: list[list]
        A list of lists that specify

    """
    ref_kw = common_reference_file_keywords(reftype="wavelengthrange",
                                            title="WFC3 TSGRISM reference file",
                                            description="WFC3 Grism-Filter Wavelength Ranges",
                                            exp_type="WFC3_TSGRISM",
                                            author=author,
                                            pupil="ANY",
                                            model_type="WavelengthrangeModel",
                                            filename=outname,
                                            )

    if wavelengthrange is None:
        # This is a list of tuples that specify the
        # order, filter, wave min, wave max
        wavelengthrange = [(-1, 'G102', 0.7488958984375, 1.1496958984375),
                           (0, 'G102', 0.7401, 1.2297),
                           (1, 'G102', 0.7496, 1.1979),
                           (2, 'G102', 0.7401, 1.1897),
                           (3, 'G102', 0.7571, 0.9878),
                           (-1, 'G141', 1.031, 1.7845),
                           (0, 'G141', 1.0402, 1.6998),
                           (1, 'G141', 0.9953, 1.7697),
                           (2, 'G141', 0.9702, 1.5903),
                           (3, 'G141', 1.0068, 1.3875),
                           (4, 'G141', 1.031, 1.7845),
                           ]

    # array of integers of unique orders
    orders = sorted(set((x[0] for x in wavelengthrange)))
    filters = sorted(set((x[1] for x in wavelengthrange)))

    # Nircam has not specified any limitation on the orders
    # that should be extracted by default yet so all are
    # included.
    if extract_orders is None:
        extract_orders = [('G102', [1]),
                          ('G141', [1]),
                          ]

    ref = wcs_ref_models.WavelengthrangeModel()
    ref.meta.update(ref_kw)
    ref.meta.exposure.p_exptype = "WFC3_TSGRISM"
    ref.meta.input_units = u.micron
    ref.meta.output_units = u.micron
    ref.wavelengthrange = wavelengthrange
    ref.extract_orders = extract_orders
    ref.order = orders
    ref.waverange_selector = filters

    history = HistoryEntry({'description': history,
                            'time': datetime.datetime.utcnow()})
    software = Software({'name': 'wfc3_reftools.py',
                         'author': author,
                         'homepage': 'https://github.com/spacetelescope/astrogrism_sandbox',
                         'version': '0.0.0'})
    history['software'] = software
    ref.history = [history]
    ref.validate()
    ref.to_asdf(outname)



def create_wfss_wavelengthrange(outname="nircam_wfss_wavelengthrange.asdf",
                                history="Ground NIRCAM Grism wavelengthrange",
                                author="STScI",
                                wavelengthrange=None,
                                extract_orders=None):
    """Create a wavelengthrange reference file for NIRCAM.

    Parameters
    ----------
    outname: str
        The output name of the file
    history: str
        History information about it's creation
    author: str
        Person or entity making the file
    wavelengthrange: list(tuples)
        A list of tuples that set the order, filter, and
        wavelength range min and max
    extract_orders: list[list]
        A list of lists that specify

    """
    ref_kw = common_reference_file_keywords(reftype="wavelengthrange",
                                            title="NIRCAM WFSS reference file",
                                            description="NIRCAM Grism-Filter Wavelength Ranges",
                                            exp_type="NRC_WFSS",
                                            author=author,
                                            pupil="ANY",
                                            model_type="WavelengthrangeModel",
                                            filename=outname,
                                            )

    if wavelengthrange is None:
        # This is a list of tuples that specify the
        # order, filter, wave min, wave max
        wavelengthrange = [(1, 'F250M', 2.500411072, 4.800260833),
                           (1, 'F277W', 2.500411072, 3.807062006),
                           (1, 'F300M', 2.684896869, 4.025318456),
                           (1, 'F322W2', 2.5011293930000003, 4.215842089),
                           (1, 'F335M', 3.01459734, 4.260432726),
                           (1, 'F356W', 3.001085025, 4.302320901),
                           (1, 'F360M', 3.178096344, 4.00099629),
                           (1, 'F410M', 3.6267051809999997, 4.5644598),
                           (1, 'F430M', 4.04828939, 4.511761774),
                           (1, 'F444W', 3.696969216, 4.899565197),
                           (1, 'F460M', 3.103778615, 4.881999188),
                           (1, 'F480M', 4.5158154679999996, 4.899565197),
                           (2, 'F250M', 2.500411072, 2.667345336),
                           (2, 'F277W', 2.500411072, 3.2642254050000004),
                           (2, 'F300M', 2.6659796289999997, 3.2997071729999994),
                           (2, 'F322W2', 2.5011293930000003, 4.136119434),
                           (2, 'F335M', 2.54572003, 3.6780519760000003),
                           (2, 'F356W', 2.529505253, 4.133416971),
                           (2, 'F360M', 2.557881113, 4.83740855),
                           (2, 'F410M', 2.5186954019999996, 4.759037127),
                           (2, 'F430M', 2.5362614100000003, 4.541488865),
                           (2, 'F444W', 2.5011293930000003, 4.899565197),
                           (2, 'F460M', 2.575447122, 4.883350419),
                           (2, 'F480M', 2.549773725, 4.899565197),
                           ]
    # array of integers of unique orders
    orders = sorted(set((x[0] for x in wavelengthrange)))
    filters = sorted(set((x[1] for x in wavelengthrange)))

    # Nircam has not specified any limitation on the orders
    # that should be extracted by default yet so all are
    # included.
    if extract_orders is None:
        extract_orders = []
        for f in filters:
            extract_orders.append([f, orders])

    ref = wcs_ref_models.WavelengthrangeModel()
    ref.meta.update(ref_kw)
    ref.meta.exposure.p_exptype = "NRC_WFSS"
    ref.meta.input_units = u.micron
    ref.meta.output_units = u.micron
    ref.wavelengthrange = wavelengthrange
    ref.extract_orders = extract_orders
    ref.order = orders
    ref.waverange_selector = filters

    history = HistoryEntry({'description': history,
                            'time': datetime.datetime.utcnow()})
    software = Software({'name': 'nircam_reftools.py',
                         'author': author,
                         'homepage': 'https://github.com/spacetelescope/jwreftools',
                         'version': '0.7.1'})
    history['software'] = software
    ref.history = [history]
    ref.validate()
    ref.to_asdf(outname)


def split_order_info(keydict):
    """
    Designed to take as input the dictionary created by dict_from_file and for
    nircam, split out and accumulate the keys for each beam/order.
    The keys must have the beam in their string, the spurious beam designation
    is removed from the returned dictionary. Keywords with the same first name
    in the underscore separated string followed by a number are assumed to be
    ranges


    Parameters
    ----------
    keydict : dictionary
        Dictionary of key value pairs

    Returns
    -------
    dictionary of beams, where each beam has a dictionary of key-value pairs
    Any key pairs which are not associated with a beam get a separate entry
    """

    if not isinstance(keydict, dict):
        raise ValueError("Expected an input dictionary")

    # has beam name fits token
    token = re.compile('^[a-zA-Z]*_(?:[+\-]){0,1}[a-zA-Z0-9]{0,1}_*')
    rangekey = re.compile('^[a-zA-Z]*_[0-1]{1,1}$')
    rdict = dict()  # return dictionary
    beams = list()

    # prefetch number of Beams, beam is the second string
    for key in keydict:
        # Separate WEDGE keys that would otherwise match beam regex
        if key[0:5] == "WEDGE":
            if "WEDGE" not in beams:
                beams.append("WEDGE")
        elif token.match(key):
            b = key.split("_")[1].upper()
            if b not in beams:
               beams.append(b)
    for b in beams:
        rdict[b] = dict()

    #  assumes that keys are sep with underscore and beam is in second section
    for key in keydict:
        # Again, reject WEDGE keys
        if key[0:5] == "WEDGE":
            b, fname = key.split("_")
            rdict[b][fname] = keydict[key]
        elif token.match(key):
            b = key.split("_")[1].upper()
            newkey = key.replace("_{}_".format(b), "_")
            rdict[b][newkey] = keydict[key]

    # look for range variables to make them into tuples
    for b, d in rdict.items():
        keys = d.keys()
        rkeys = []
        odict = {}
        for k in keys:
            if rangekey.match(k):
                rkeys.append(k)
        for k in rkeys:
            mlist = [m for m in rkeys if k.split("_")[0] in m]
            root = mlist[0].split("_")[0]
            if root not in odict:
                for mk in mlist:
                    if eval(mk[-1]) == 0:
                        zero = d[mk]
                    elif eval(mk[-1]) == 1:
                        one = d[mk]
                    else:
                        raise ValueError("Unexpected range variable {}"
                                         .format(mk))
                odict[root] = (zero, one)
        # combine the dictionaries and remove the old keys
        d.update(odict)
        for k in rkeys:
            del d[k]

    return rdict


def dict_from_file(filename):
    """Read in a file and return a named tuple of the key value pairs.

    This is a generic read for a text file with the line following format:

    keyword<token>value

    Where keyword should start with a character, not a number
    Non-alphabetic starting characters are ignored
    <token> can be space or comma

    Parameters
    ----------
    filename : str
        Name of the file to interpret

    Examples
    --------
    dict_from_file('NIRCAM_C.conf')

    Returns
    -------
    dictionary of deciphered keys and values

    """
    token = '\s+|(?<!\d)[,](?!\d)'
    letters = re.compile("(^[a-zA-Z])")  # starts with a letter
    numbers = re.compile("(^(?:[+\-])?(?:\d*)(?:\.)?(?:\d*)?(?:[eE][+\-]?\d*$)?)")
    empty = re.compile("(^\s*$)")  # is a blank line

    print("\nReading {0:s}  ...".format(filename))
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    content = dict()
    for line in lines:
        value = None
        key = None
        if not empty.match(line):
            if letters.match(line):
                pair = re.split(token, line.strip())
                if len(pair) == 2:  # key and value exist
                    key = pair[0]  # first item is the key
                    val = pair[1]  # second item is the value
                    if letters.match(val):
                        value = val
                    if numbers.fullmatch(val):
                        value = eval(val)
                if len(pair) == 3:  # key min max exist
                    key = pair[0]
                    val1, val2 = pair[1:]
                    if numbers.fullmatch(val1) and numbers.fullmatch(val2):
                        value = (eval(val1), eval(val2))
                    else:
                        raise ValueError("Min/max values expected for {0}"
                                         .format(key))
                elif len(pair) > 3: # Key and many values exist (DISPX and DISPY)
                    key = pair[0]
                    value = []
                    for i in range(1, len(pair)):
                        value.append(eval(pair[i]))
                    value = tuple(value)

        # ignore the filter file pointings and the sensitivity files these are
        # used for simulation
        if key and (value is not None):
            if (("FILTER" not in key) and ("SENSITIVITY" not in key)):
                content[key] = value
                print("Setting {0:s} = {1}".format(key, value))

    return content
