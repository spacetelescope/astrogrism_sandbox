# Flat Fielding of Slitless Spectroscopy

### Participants: Kornpob Bhirombhakdi, [Russell Ryan](mailto:rryan@stsci.edu), and [Mehdi Rezaie](mailto:mr095415@ohio.edu)
### Astrogrism-27 Goal
***The goal of this ticket is to assess what it might take to migrate away from aXe for this step of grism data processing (flat fielding).  This notebook has benefited significantly from Astrogrism-53.***


## 0.  Primer
Slitless spectroscopy uses a dispersive and transmissive optical element in the collimated beam to create a complete spectroscopic view of any astrophysical scene.  However, the lack of any light-restricting aperture (such as a slitlet, fiber, etc.), means that the spectral trace of multiple sources may overlap in the two-dimensional grism image.  Therefore a given imaging pixel may record the flux (at different wavelengths) from the multiple sources and/or background components (such as Zodiacal background, thermal emission, etc.).  Here, we discuss the issues associated with extending the classical application of a flat-field image to correct for pixel-to-pixel variations in the imaging device to that of slitless spectroscopy.


This notebook was prepared by Kornpob Bhirombhakdi, [Russell Ryan](mailto:rryan@stsci.edu), and [Mehdi Rezaie](mailto:mr095415@ohio.edu) as part of the Astrogrism coding spring May 11-16, 2020.  


## 1.  Review of Key Concepts

optical path of an imaging device may include physical obstructions (e.g. dust or particulates on optimal elements), illumination effects (e.g. vignetting), and/or variable pixel-to-pixel sensitivities, which leads to a nonuniform response.


Since the sensitivity of a given pixel depends on the wavelength of incident light, the flat-field response must be characterized as a (${\cal F}(\lambda)$), and therefore the measured response in a given pixel $(x,y)$ is given as an integral:
$$
r_{x,y}  = \int {\cal F}_{x,y}(\lambda)\, f_{x,y}(\lambda)\,\mathrm{d}\lambda
\tag{1}
$$
where $f(\lambda)$ is the spectrum of the incident light.  


## 2.  Extensions to Slitless Spectroscopy

One key difference between a grism optical element and a standard imaging element is that light incident on a given pixel $(x_d,y_d)$ will be recorded a new pixel position $(x_g,y_g)$, where that remapping is done as a function of wavelength and $(x_d,y_d)$.  This process must be calibrated by standard observations, which often results in determining the spectral trace and dispersion solution.  Software has been developed by [Nor Prizkal and Russell Ryan](https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2017/WFC3-2017-01.pdf) to implement these transformations, and are represented a parametric functions:


$$
x_g = \mathrm{X} (x_d,y_d,t) + x_d\\
y_g = \mathrm{Y} (x_d,y_d,t) + y_d\\
\lambda = \Lambda (x_d,y_d,t)
\tag{2}
$$

where $t$ is a parameter such that $0\leq t\leq1$.  With that, it is apparent that the response of a pixel (eq. 1) must be reconsidered as the incident spectrum effectively eminates from a different position:
$$
r_{x_g,y_g} = \int {\cal F}_{x_g,y_g}(\lambda)\,f_{x_d,y_d}(\lambda)\,\mathrm{d}\lambda.
\tag{3}
$$
***Note, here we take care to explicitly subscript $(x,y)$ with either $g$ or $d$ to refer to the measured position in a grism image or effective position from a direct image, respectively.***  Therefore, when flat-fielding one must invert the family of functions for $(x_g,y_g,\lambda)$ to account for the relevant spectral components (whether from astrophysical sources or background contributions).  



The flat-field image for slitless spectroscopy is often characterized as a polynomial in wavelength, where each of the polynomial coefficients is a two-dimensional image:
$$
{\cal F}_{x,y}(\lambda) = \sum_i {\cal F}_{x,y,i}\left(\frac{\lambda-\lambda_0}{\lambda_1-\lambda_0}\right)^i
\tag{4}
$$
where $\lambda_0$ and $\lambda_1$ are arbitrary values that are often choosen to be approximately the blue and red edges (respectively) of the throughput of the grism element. In common usage, this is often referred to as a *flatfield cube*.  

Here is a schematic image of a flat field cube.

![flat field](notebooks/figures/flatfieldcube.png)


The required modules are:
1. astropy (https://www.astropy.org)
2. numpy (https://numpy.org)
3. gwcs (https://gwcs.readthedocs.io/en/latest/)
4. scipy
5. axehelper
6. polynomial2d




```Python
# FlatField takes the paths to flat field cube and config file
# It also takes a set of reference coordinates


conf_file = '../inputs/aXe_config/ACS.WFC.CHIP1.Cycle13.5.conf'
flat_file = '../inputs/ACS_WFC_Grism_Data/save/CONF/WFC.flat.cube.CH1.2.fits'


xref = np.array([400.]) # KP: assume source location
yref = np.array([500.]) # (with other calibration) leading xref,yref

ff = FlatField(flat_file, conf_file)
ff.run(xref, yref)

flatfield = ff.output['flatfield'] # is a numpy array but also has the imshow functionality
flatfield.imshow(origin='lower', cmap='viridis', hold=True)
```
![](notebooks/figures/flatfield.png)
