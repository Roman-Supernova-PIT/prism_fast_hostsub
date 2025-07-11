from astropy.modeling import models
import numpy as np
import matplotlib.pyplot as plt
from romanisim import psf

s1 = models.Sersic1D(1.0, 2.0, 1)
g1 = models.Gaussian1D(5.0, 0, 2.0)

model = s1 + g1

x = np.linspace(-5, 5, 200)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, model(x))

plt.show()

# ======
# Get Roman PSF
avg_psf = psf.make_psf(sca=1, filter_name='F129',
                       pix=(2048, 2048))

# numpy array of the PSF
avg_psf_arr = avg_psf.image.array
