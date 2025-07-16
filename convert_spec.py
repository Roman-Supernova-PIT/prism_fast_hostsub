import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import astropy.units as u

import sys
import os

home = os.getenv('HOME')
prismdir = hostsubdir = home + '/Documents/Roman/PIT/prism/'
hostsubdir = prismdir + 'hostlight_subtraction/'
datadir = hostsubdir + 'simdata_prism_galsn/'

c_ang = constants.speed_of_light * 1e10  # A/s; default in m/s

# Read in the flam spectrum
sn_spec_fname = datadir + 'lfnana.txt'
sn_wav, sn_flam = np.loadtxt(sn_spec_fname, unpack=True)

# Now convert to fnu
fnu = np.zeros(len(sn_flam))
for w in range(len(sn_wav)):
    wav = sn_wav[w]
    fnu[w] = (wav**2 / c_ang) * sn_flam[w]

np.savetxt(sn_spec_fname.replace('.txt', '_fnu.txt'),
           X=np.c_[sn_wav, fnu],
           fmt='%10.4e')

# Conversion using astropy to check
flam_quantity = sn_flam * u.erg / u.cm**2 / u.s / u.AA
print(flam_quantity)
fnu_quantity = flam_quantity.to(u.Jy,
                                equivalencies=u.spectral_density(sn_wav*u.AA))
print(fnu_quantity)

# Plot spectrum in fnu
fig = plt.figure()
ax = fig.add_subplot(111)
offset = 0.1
ax.plot(sn_wav, fnu / 1e-29 + offset)  # in micro-Janskys
ax.plot(sn_wav, fnu_quantity/1e-6)  # in micro-Janskys
ax.set_xlim(2000, 20000)
plt.show()

sys.exit(0)
