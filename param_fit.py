import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import astropy
from astropy.wcs import WCS
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

import sys
import os
# from tqdm import tqdm

home = os.getenv('HOME')
prismdir = hostsubdir = home + '/Documents/Roman/PIT/prism/'
hostsubdir = prismdir + 'hostlight_subtraction/'
datadir = hostsubdir + 'simdata_prism_galsn/'

x1d_dir = home + '/Documents/Roman/prism_quick_reduction/romanprism-fast-x1d/'
sys.path.append(x1d_dir)
import romanprism_fast_x1d as oned_utils  # noqa


def get_model_init(model, y_fit, x_fit, xloc):

    if model == 'sersic':

        # Initial guesses
        amplitude_init = y_fit.max()
        r_init = 10  # half-light radius; pixels
        # init sersic index
        # 1 is exponential; 4 is de vaucouleurs; 0.5 is gaussian
        n_init = 4

        # print('Initial guesses for Sersic profile:')
        # print('r-eff (half-light rad; pix):', r_init)
        # print('n [Sersic index]:', n_init)

        model_init = models.Sersic1D(amplitude=amplitude_init, r_eff=r_init,
                                     n=n_init)

    elif model == 'moffat':

        # Initial guess: amplitude=max(y), x_0 at peak, gamma=1, alpha=1.5
        amplitude_init = y_fit.max()
        x_0_init = xloc
        gamma_init = 5
        alpha_init = 1.5

        # print('Initial guesses for Moffat profile:')
        # print('x0:', x_0_init)
        # print('gamma [width of distribution]:', gamma_init)
        # print('alpha [power index]:', alpha_init)

        model_init = models.Moffat1D(amplitude=amplitude_init, x_0=x_0_init,
                                     gamma=gamma_init, alpha=alpha_init,
                                     fixed={'x_0': True},
                                     bounds={'amplitude': (0, amplitude_init)})

    elif model == 'gauss':

        # Initial guesses
        amplitude_init = y_fit.max()
        mean_init = x_fit[np.argmax(y_fit)]
        std_init = 10

        model_init = models.Gaussian1D(amplitude=amplitude_init,
                                       mean=mean_init,
                                       stddev=std_init)

    return model_init


def prep_fit(y, x=None, mask=None):

    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    # Apply mask
    if mask is not None and not donotmask:
        x_fit = np.delete(x, mask)
        y_fit = np.delete(y, mask)
    else:
        x_fit = x
        y_fit = y

    # Additionally mask NaNs and negative values
    valid_mask = (y_fit >= 0) & (np.isfinite(y_fit))
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    # Also check min len requirement
    prep_flag = True
    if len(x_fit) < numpix_fit_thresh:
        prep_flag = False

    return x_fit, y_fit, prep_flag


def fit_1d(y_fit, x_fit, xloc=50, model=None, row_idx=None):
    """
    Fit a Moffat or Sersic profile to 1D data using astropy.

    Parameters:
        y (array-like): 1D data values.
        x (array-like, optional): x values. If None, uses np.arange(len(y)).

    Returns:
        fitted_model: The best-fit astropy model.
    """

    model_init = get_model_init(model, y_fit, x_fit, xloc)

    fit = fitting.TRFLSQFitter()
    try:
        fitted_model = fit(model_init, x_fit, y_fit)
    except astropy.modeling.fitting.NonFiniteValueError:
        print(x_fit)
        print(y_fit)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_fit, y_fit, 'o', markersize=4, color='k')
        ax.set_title('Fitting failed due to non-finite values.\n' +
                     'Row index: ' + str(row_idx))
        plt.show()
        sys.exit(1)

    return fitted_model


def get_contiguous_slices(arr, min_length=10):
    """
    Finds the start and end indices of contiguous True
    slices of at least min_length.
    Returns a list of (start, end) tuples (end is inclusive).
    """

    result = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            start = i
            while i < n and arr[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_length:
                result.append((start, end))
        else:
            i += 1

    return result


def get_row_std_objmasked(arr, mask=None):
    if mask is not None:
        arr_fit = np.delete(arr, mask)
        std = np.std(arr_fit)
        return std
    else:
        return np.std(arr)


def get_sn_host_loc(fname, snra, sndec, hostra, hostdec):

    header = fits.getheader(datadir + fname, ext=1)
    wcs = WCS(header)

    snloc = wcs.world_to_pixel_values(snra, sndec)
    xsn = float(snloc[0])
    ysn = float(snloc[1])

    hostloc = wcs.world_to_pixel_values(hostra, hostdec)
    xhost = float(hostloc[0])
    yhost = float(hostloc[1])

    return xsn, ysn, xhost, yhost


if __name__ == '__main__':

    # ==========================
    # START USER INPUTS
    showcutoutplot = False
    showfit = False

    # use this flag to not mask anything
    # we need this if masking one object would also mask the other
    donotmask = False

    # Minimum number of pixels in a row above 3-sigma to fit a profile
    # These have to be contiguous
    # i.e., these are sigma_thresh above the background
    numpix_fit_thresh = 5

    # Sigma threshold for the pixels to be considered
    sigma_thresh = 2

    # user guess for starting row at which real signal
    # for the host galaxy starts
    start_row = 50

    # starting and ending wavelengths for x1d
    start_wav = 0.7
    end_wav = 1.8

    # Pads for masking SN and host
    galmaskpad = 6
    snmaskpad = 2

    # Cutout size
    cutoutsize_y_lo = 200
    cutoutsize_y_hi = 100
    cutoutsize_x = 100

    spec_img_exptime = 900  # seconds

    # File name
    fname = 'test_prism_WFI_rollAngle000_dither0.fits'

    # Coordinates
    snra = 7.6022277
    sndec = -44.7897423
    hostra = 7.602432752
    hostdec = -44.78963685

    # Input spectra
    host_spec_fname = datadir + 'lfgal.txt'
    sn_spec_fname = datadir + 'lfnana.txt'

    # For 1D extraction
    obj_one_sided_width = 2

    # END USER INPUTS
    # ==========================

    print('Working on file:', fname)

    # Get SN and host locations from truth file
    xsn, ysn, xhost, yhost = get_sn_host_loc(fname, snra, sndec,
                                             hostra, hostdec)

    # Read in input spectra
    host_input_wav, host_input_flux = np.loadtxt(host_spec_fname, unpack=True)
    sn_input_wav, sn_input_flux = np.loadtxt(sn_spec_fname, unpack=True)

    # Load image
    prismimg = fits.open(datadir + fname)
    # ext=1 is SCI
    # ext=2 is ERR
    # ext=3 is DQ
    # ext=4 is TRUE
    prismdata = prismimg[4].data

    # convert to integer pixels
    row = int(ysn)
    col = int(xsn)
    # print('SN row:', row, 'col:', col)

    # TODO list
    print('\nTODO LIST:')
    print('* NOTE: Automate the cropping later. You will need galaxy',
          'half-light radius or the a & b elliptical axes from',
          'something like SExtractor to measure the cutout size.')
    print('* NOTE: Using default half-light radius initial guess.',
          'This would be better if user provided.')
    print('* NOTE: Using only Sersic/Gaussian profile to fit galaxy.',
          'This really should be a model profile convolved with',
          'lambda dependent PSF. Convolve Moffat for SN with PSF too.')
    print('* NOTE: approx host galaxy position used here for masking.',
          'This needs to come from the user.')
    print('TODO: write code to handle case where host galaxy is also',
          'contaminated by another galaxy spectrum (typically only a part',
          'of the host spec will be covered by the other spectrum).')
    print('* NOTE: using the max val in the cutout as the amplitude',
          'as the initial guess. Should work in most cases.')
    print('* NOTE: SN 1D spectrum is just collapsing the 2D residuals.',
          'It calls onedutils which does the simple sum.',
          'This should use something like the Horne86 optimal extraction.')
    print('* NOTE: try stacking rows for the harder cases.')
    print('* NOTE: iterate with constraints on fit params.')
    print('* NOTE: Check WCS and x,y coords returned by above func in ds9.')
    print('* NOTE: Figure out how to handle ERR and DQ extensions.')
    print('\n\n')

    # Cutout centered on SN loc
    cutout = prismdata[row - cutoutsize_y_lo: row + cutoutsize_y_hi,
                       col - int(cutoutsize_x/2): col + int(cutoutsize_x/2)]

    # plot
    if showcutoutplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(np.log10(cutout), origin='lower', vmin=1.5, vmax=2.5)
        cbar = fig.colorbar(cax)
        cbar.set_label('log(pix val)')
        plt.show()
        fig.savefig(fname.replace('.fits', '_cutout.png'), dpi=200,
                    bbox_inches='tight')
        fig.clear()
        plt.close(fig)

    # ==========================
    # For diagnostic purposes
    # The end_row should be set to the end of the cutout
    # but this can be set to the start_row + some other
    # number of rows that you'd like to see fits for
    end_row = cutout.shape[0]

    # Now fit the profile
    xarr = np.arange(cutoutsize_x)

    # We actually need to create the masks first
    # mask the SN
    sn_mask_idx = np.arange(int(cutoutsize_x/2) - snmaskpad,
                            int(cutoutsize_x/2) + snmaskpad)
    # mask the galaxy
    cutout_host_x = int(cutoutsize_x/2) + int(int(xhost) - col)
    # print('Host cutout center idx:', cutout_host_x)
    gal_mask_idx = np.arange(cutout_host_x - galmaskpad,
                             cutout_host_x + galmaskpad)

    combinedmask = np.union1d(sn_mask_idx, gal_mask_idx)
    # print('combined mask:', combinedmask)

    # Empty array for host model
    host_model = np.zeros_like(cutout)

    # loop over all rows
    for i in range(start_row, end_row):
        # print('\nFitting row:', i)
        profile_pix = cutout[i]

        # ----- Skipping criterion
        stdrow = get_row_std_objmasked(profile_pix, mask=combinedmask)
        # print('Estimated stddev (with masked SN and host locations)',
        #       'of this row of pixels:', stdrow)
        # contiguous pixels over 3-sigma
        pix_over_thresh = (profile_pix > sigma_thresh*stdrow)  # boolean
        res = get_contiguous_slices(pix_over_thresh,
                                    min_length=numpix_fit_thresh)
        if not res:
            print('Row:', i, 'Too few pixels above 3-sigma to fit. Skipping.')
            continue

        # ------ Proceed to fitting
        # Fit a "modified" Sersic profile to the galaxy
        # print('\nFitting galaxy profile...')
        # print('SN mask:', sn_mask_idx)
        gal_x_fit, gal_y_fit, gal_prep_flag = prep_fit(profile_pix, xarr,
                                                       mask=sn_mask_idx)
        if not gal_prep_flag:
            continue
        galaxy_fit = fit_1d(gal_y_fit, gal_x_fit, xloc=cutout_host_x,
                            model='moffat', row_idx=i)

        # Fit a moffat profile to the supernova "residual"
        # generate mask first. You know the x loc of the SN
        # we are just going to mask 10 pix on each side.
        # Note that these indices are relative to the cutout indices.
        # because the cutout was already centered on xsn we can just
        # mask out hte central pixels
        # print('\nFitting SN profile...')
        residual_pix = profile_pix - galaxy_fit(xarr)

        # Now mask the galaxy indices
        # This is needed even though for an ideal fit
        # the residuals should be normally distributed around zero
        # for the host galaxy. IT is needed because the fits arent
        # usually perfect.
        # print('galaxy mask:', gal_mask_idx)
        sn_x_fit, sn_y_fit, sn_prep_flag = prep_fit(residual_pix, xarr,
                                                    mask=gal_mask_idx)
        if not sn_prep_flag:
            continue
        moffat_fit = fit_1d(sn_y_fit, sn_x_fit, model='moffat')

        if showfit:
            fig = plt.figure(figsize=(7, 5))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            # plot points and fit
            ax1.plot(xarr, profile_pix, 'o', markersize=4, color='k')
            # ax1.set_yscale('log')
            ax1.plot(xarr, moffat_fit(xarr) + galaxy_fit(xarr), color='r',
                     label='Full fit \n' + 'Row: ' + str(i))
            ax1.legend(loc=0)

            # plot galaxy residuals
            ax2.scatter(xarr, profile_pix - galaxy_fit(xarr), s=5, color='k',
                        label='Residuals after removing galaxy fit')
            ax2.legend(loc=0)
            # plot SN mask
            ax2.axvspan(sn_mask_idx[0], sn_mask_idx[-1], alpha=0.3,
                        color='gray')

            # plot SN residuals
            ax3.scatter(xarr, profile_pix - moffat_fit(xarr), s=5, c='k',
                        label='Residuals after removing SN fit')
            ax3.legend(loc=0)
            # plot galaxy mask
            ax3.axvspan(gal_mask_idx[0], gal_mask_idx[-1], alpha=0.3,
                        color='gray')

            plt.show()
            fig.clear()
            plt.close(fig)

        # Get fit params
        host_fit_paramnames = galaxy_fit.param_names

        # Save the host fit model
        host_model_row = galaxy_fit(xarr)
        host_model[i] = host_model_row

    # ==========================
    # Get the SN spectrum
    recovered_sn_2d = cutout - host_model

    imghdr = prismimg[0].header
    wcs = WCS(imghdr)
    # WCS unused if coordtype is 'pix'.
    # The object X, Y are the center of the cutout
    # and the 1.55 micron row index.
    obj_x = int(cutoutsize_x/2)
    obj_y = cutout.shape[0] - cutoutsize_y_hi
    # print(obj_x, obj_y)
    rs, re, cs, ce, specwav = oned_utils.get_bbox_rowcol(obj_x, obj_y, wcs,
                                                         obj_one_sided_width,
                                                         coordtype='pix',
                                                         start_wav=start_wav,
                                                         end_wav=end_wav)

    print('Row start:', rs, 'Row end:', re)
    print('Col start:', cs, 'Col end:', ce)
    # print('Wavelengths:', specwav)
    # print(specwav.shape)

    # Collapse to 1D
    # Since we know the location of the SN, we will just
    # use a few pixels around it.
    spec2d = recovered_sn_2d[rs: re + 1, cs: ce + 1]
    sn_1d_spec = np.nanmean(spec2d, axis=1)

    # Read in the effective area curve for the prism
    # these areas are in square meters
    roman_effarea = np.genfromtxt(prismdir + 'Roman_effarea_20201130.csv',
                                  dtype=None, names=True, delimiter=',')

    prism_effarea = roman_effarea['SNPrism'] * 1e4  # cm2
    prism_effarea_wave = roman_effarea['Wave'] * 1e4  # angstroms

    # covert to physical units
    sn_1d_spec_phys = oned_utils.convert_to_phys_units(spec2d, specwav, 'DN',
                                                       prism_effarea_wave,
                                                       prism_effarea,
                                                       subtract_bkg=False,
                                                       spec_img_exptime=1000)

    # ==========
    # Show all host subtraction
    fig = plt.figure(figsize=(6, 6))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(np.log10(cutout), origin='lower', vmin=0.5, vmax=2.5)
    ax1.set_title('SN + Host original image cutout')

    ax2.imshow(np.log10(host_model), origin='lower', vmin=0.5, vmax=2.5)
    ax2.set_title('File: ' + fname + '\n' + 'Host model')

    ax3.imshow(np.log10(recovered_sn_2d), origin='lower',
               vmin=0.5, vmax=2.5)
    ax3.set_title('SN residual')

    fig.savefig(fname.replace('.fits', '_2dparamfit.png'),
                dpi=200, bbox_inches='tight')
    fig.clear()
    plt.close(fig)

    # ==========
    # Now plot input and recovered spectra
    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(10, 5, hspace=0.05, wspace=0.05,
                  top=0.95, bottom=0.1, left=0.1, right=0.95)
    ax1 = fig.add_subplot(gs[:7, :])
    ax2 = fig.add_subplot(gs[7:, :])

    ax2.set_xlabel('Wavelength [microns]')
    ax1.set_ylabel('Flam [erg/s/cm2/Angstrom]')
    ax2.set_ylabel('Residuals')

    host_smaller_cutout = host_model[:, cutout_host_x - 25:cutout_host_x + 25]
    host_rec_flux = np.mean(host_smaller_cutout, axis=1)

    sn_input_wav_microns = sn_input_wav/1e4

    ax1.plot(specwav, sn_1d_spec_phys, '-',
             color='crimson', lw=1.5, label='Recovered SN spec [flam]')
    ax1.plot(sn_input_wav_microns, sn_input_flux, '-',
             color='mediumseagreen', label='Input SN spec')

    ax1.legend(loc=0)

    ax1.set_xlim(0.65, 2.0)
    ax1.set_ylim(0, 1.5e-19)

    ax1.set_xticklabels([])

    # plot residuals
    sn_input_flux_inwav = griddata(points=sn_input_wav_microns,
                                   values=sn_input_flux,
                                   xi=specwav)
    spec_resid = (sn_input_flux_inwav - sn_1d_spec_phys) / sn_input_flux_inwav

    ax2.scatter(specwav, spec_resid, s=4, c='k')

    ax2.set_xlim(0.65, 2.0)
    ax2.set_ylim(-1.5, 1.5)

    fig.savefig(fname.replace('.fits', '_1dparamfit_phys.png'),
                dpi=200, bbox_inches='tight')
    # plt.show()

    # Close image
    prismimg.close()

    sys.exit(0)
