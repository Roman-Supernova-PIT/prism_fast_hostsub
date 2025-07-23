import numpy as np
from scipy.interpolate import interp1d
import synphot as syn
from astropy import units as u
from astropy import coordinates
from astropy.io import fits

from ilia import constants

import sys
import os

home = os.getenv('HOME')
hostsubdir = home + "/Documents/Roman/PIT/prism/hostlight_subtraction/"
datacubeFilename = hostsubdir + "fromMassimo/gal_cube_os3_3D.fits"

DEF_COORDS = coordinates.SkyCoord(7.60244299, -44.79116827,
                                  frame='icrs', unit='deg')
ROMAN_PIXSCALE = constants.ROMAN_PIXSCALE
SN_x = 95
SN_y = 70
scale_factor = 10  # scaling factor for SN flux

# =======
# Open datacube
hduList = fits.open(datacubeFilename)
datacube = hduList[0].data[:, 0:-1, 0:-1] * syn.units.FNU

oversample = int(hduList[0].header['OVERSAMP'])
lambdaMin = float(hduList[0].header['LBDAMIN'])
lambdaMax = float(hduList[0].header['LBDAMAX'])
lambdaStep = float(hduList[0].header['LBDASTEP'])

wavelength = (np.arange(lambdaMin, lambdaMax+0.1*lambdaStep, lambdaStep)
              * u.angstrom)

redshift = hduList[0].header['REDSHIFT']

print(datacube.shape, wavelength.shape)

# load SN SED
SN = np.loadtxt(hostsubdir + 'simdata_prism_galsn/lfnana_fnu.txt')
SN_SED = interp1d(SN[:, 0], SN[:, 1], kind='linear')

datacube[:, SN_y - 1, SN_x - 1] += (scale_factor
                                    * SN_SED(wavelength) * syn.units.FNU)

sys.exit(0)
