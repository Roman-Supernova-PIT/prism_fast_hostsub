from astropy.modeling import models, fitting, Fittable2DModel
from astropy.convolution import Model2DKernel, convolve_models
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from romanisim import psf

from param_fit import get_sn_host_loc, get_all_masks

import sys
import os
home = os.getenv('HOME')


def get_filter_psf_arr(filtername):

    avg_psf = psf.make_psf(sca=1, filter_name=filtername,
                           pix=(2048, 2048))
    # numpy array of the PSF
    avg_psf_arr = avg_psf.image.array

    return avg_psf_arr


print('\nNOTE: Only broadband PSFs available through romanisim.\n',
      'Using PSF constructed by Massimo for the prism.',
      'Switch back to romanisim when prism PSF is available there.')

# ======
# Get Roman PSF from Tri
hostsubdir = home + '/Documents/Roman/PIT/prism/hostlight_subtraction/'
ilia_datadir = hostsubdir + 'fromMassimo/ilia-config/Roman/'
psf_fname = 'Roman_PSF_datacube_650_1900_26_fov15_oversample5_SCA01.fits'
prism_psf_fname = ilia_datadir + psf_fname
assert os.path.isfile(prism_psf_fname)

# ======
# Get Roman PSF for F129 and compare to the prism PSF average around 1.29um
prism_psf_hdu = fits.open(prism_psf_fname)
print(prism_psf_hdu.info())

# Get the F129 PSF
f129_psf = get_filter_psf_arr('F129')

# ======
# Test code block to make model and fit

# First get some test data
data_fname = 'test_prism_WFI_rollAngle060_dither0.fits'
datadir = hostsubdir + 'simdata_prism_galsn/'
prismimg = fits.open(datadir + data_fname)
prismdata = prismimg[1].data

# get cutout size config
cs_x = 100
cs_y_lo = 200
cs_y_hi = 100

# Convert SN and host locations from user to x,y
# Get user coords
# Coordinates
snra = 7.60222713
sndec = -44.79127394
hostra = 7.60242200
hostdec = -44.79115301
xsn, ysn, xhost, yhost = get_sn_host_loc(data_fname, snra, sndec,
                                         hostra, hostdec)

# convert to integer pixels
row = int(ysn)
col = int(xsn)
# print('SN row:', row, 'col:', col)

# Cutout centered on SN loc
cutout = prismdata[row - cs_y_lo: row + cs_y_hi,
                   col - int(cs_x/2): col + int(cs_x/2)]

# Get host location within cutout
cutout_host_x = int(cs_x/2) + int(int(xhost) - col)

xarr = np.arange(cs_x)

for i in range(30, cutout.shape[0]):

    # fit thresh
    sigma_thresh = 2.5
    numpix_fit_thresh = 5

    profile_pix = cutout[i]


# Now create model and fit
amplitude_init = 1
x_0_init = 1
gamma_init = 1
alpha_init = 1
model_init = models.Moffat1D(amplitude=amplitude_init, x_0=x_0_init,
                             gamma=gamma_init, alpha=alpha_init,
                             fixed={'x_0': True},
                             bounds={'amplitude': (0, amplitude_init),
                                     'gamma': (0, 10),
                                     'alpha': (0, 10)})
# convolve with PSF
f129_psf_model = Fittable2DModel()
prismpsf = Model2DKernel(f129_psf_model)
conv_model = convolve_models(model_init, prismpsf)

sys.exit(0)

fit = fitting.TRFLSQFitter()

model_nopsf = fit(model_nopsf_init, xfit, yfit)
model_withjbandpsf = fit(model_withjbandpsf_init, xfit, yfit)
model_withprismpsf = fit(model_withprismpsf_init, xfit, yfit)

# PLot to show fitting with and without prism PSF
# also shows fit if we use the broadband PSF for F129
# instead of the prism PSF
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, model(x))

plt.show()

sys.exit(0)
