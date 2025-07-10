import numpy as np
from scipy import constants

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

sys.exit(0)
