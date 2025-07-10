import numpy as np
from astropy.modeling import models
import matplotlib.pyplot as plt
import matplotlib as mpl

from covmat import boxsmooth
import sys


def boxsmooth_v2(arr, widx, widy):

    # Shape of original image
    nrow, ncol = arr.shape

    # Construct empty new smoothed image
    smootharr = np.zeros((nrow-2*widy, ncol-2*widx))

    # Loop
    for col in range(ncol - widx):
        for row in range(nrow - widy):

            # Inner loop for box window average
            boxsum = 0
            for bcol in range(-widx, widx+1):
                for brow in range(-widy, widy+1):
                    boxsum += arr[row + brow, col + bcol]

            boxmean = boxsum / (2*widx+1) / (2*widy+1)
            smootharr[row - widy, col - widx] = boxmean

    return smootharr


def test_numpy_roll(arr, Np=10):

    mpl.rcParams['text.usetex'] = False  # tex slows things down

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(121)
    ax1.set_title('Original Image')
    ax2 = fig.add_subplot(122)

    count = 0
    for dc in range(Np):  # column shift loop
        for dr in range(1-Np, Np):  # row loop, incl negatives

            # ism = arr*np.roll(np.roll(arr, -dc, axis=0), -dr, axis=1)
            if dr < 0 and dc == 0:
                # i.e, inner roll won't happen
                # why is this skipped though
                continue

            ism = np.roll(np.roll(arr, -dc, axis=1), -dr, axis=0)
            print('-dc, -dr:', -dc, -dr)

            ax1.imshow(arr, cmap='gray_r', origin='lower',
                       vmin=-2, vmax=25)
            ax2.imshow(ism, cmap='gray_r', origin='lower',
                       vmin=-2, vmax=25)
            ax2.set_title('Rolled Image \n'
                          + '-dc:' + str(-dc)
                          + '  -dr:' + str(-dr))

            plt.pause(0.2)

            count += 1
            if count > 50:
                sys.exit(0)

    return None


if __name__ == '__main__':

    ref = np.array(([117,  216,  315,  414,  513,  612,  711,  810,  909],
                    [126,  225,  324,  423,  522,  621,  720,  819,  918],
                    [135,  234,  333,  432,  531,  630,  729,  828,  927],
                    [144,  243,  342,  441,  540,  639,  738,  837,  936],
                    [153,  252,  351,  450,  549,  648,  747,  846,  945],
                    [162,  261,  360,  459,  558,  657,  756,  855,  954],
                    [171,  270,  369,  468,  567,  666,  765,  864,  963],
                    [180,  279,  378,  477,  576,  675,  774,  873,  972],
                    [189,  288,  387,  486,  585,  684,  783,  882,  981]))

    arr = np.arange(1, 122).reshape(11, 11).T

    sa, tot = boxsmooth(arr, 3, 3)

    print(sa)
    print(sa.shape, ref.shape)
    print(np.array_equal(sa, ref))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray_r', origin='lower', vmin=1, vmax=300)
    ax.imshow(ref, cmap='gray_r', origin='lower', vmin=1, vmax=300)
    plt.show()

    sys.exit(0)

    # Example usage
    test_arr_size = 200  # square

    # Start with a baseline noise
    noisearr = np.random.normal(loc=0, scale=1,
                                size=(test_arr_size, test_arr_size))

    # Now add a Gaussian peak
    g2d = models.Gaussian2D()
    xgrid, ygrid = np.meshgrid(np.arange(test_arr_size),
                               np.arange(test_arr_size))

    # location
    x0, y0 = 50, 50
    # Standard deviation of the Gaussian
    xsigma = 2
    ysigma = 2.5
    amplitude = 20
    theta = 0.5  # Rotation angle in radians. Clockwise

    # Add a few sources
    gp1 = g2d.evaluate(xgrid, ygrid, amplitude, x0, y0,
                       xsigma, ysigma, theta)
    gp2 = g2d.evaluate(xgrid, ygrid, amplitude + 10, x0 + 25, y0 + 25,
                       xsigma + 4, ysigma, theta)
    gp3 = g2d.evaluate(xgrid, ygrid, amplitude - 10, x0 + 100, y0 + 45,
                       xsigma + 1, ysigma + 1, theta)

    arr = gp1 + gp2 + gp3 + noisearr

    widx = 2
    widy = 2

    cloudcov_smoothed_arr, total = boxsmooth(arr, widx, widy)
    print('CloudCov smoothed array shape:', cloudcov_smoothed_arr.shape)

    manual_smoothed_arr = boxsmooth_v2(arr, widx, widy)
    print('BAJ smoothed array shape:', manual_smoothed_arr.shape)

    # Plot the original and smoothed arrays side by side
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(arr, cmap='gray_r', origin='lower',
               vmin=-2, vmax=30)
    ax1.set_title('Original Image')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(manual_smoothed_arr, cmap='gray_r', origin='lower',
               vmin=-2, vmax=30)
    ax2.set_title('BAJ Boxcar Smoothed Image')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(cloudcov_smoothed_arr, cmap='gray_r', origin='lower',
               vmin=-2, vmax=30)
    ax3.set_title('CloudCovFix Boxcar Smoothed Image')

    plt.show()
    fig.clear()
    plt.close(fig)
    sys.exit(0)

    test_numpy_roll(arr)




    sys.exit(0)
