import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import astropy
from astropy.wcs import WCS
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

import sys
import os
# from tqdm import tqdm
from pprint import pprint
import yaml

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
                                     bounds={'amplitude': (0, amplitude_init),
                                             'gamma': (0, 10),
                                             'alpha': (0, 10)})

    elif model == 'gauss':

        # Initial guesses
        amplitude_init = y_fit.max()
        mean_init = x_fit[np.argmax(y_fit)]
        std_init = 10

        model_init = models.Gaussian1D(amplitude=amplitude_init,
                                       mean=mean_init,
                                       stddev=std_init)

    return model_init


def prep_fit(cfg, y, x=None, mask=None):
    """
    Function to prepare x,y arrays for 1D model fitting.
    It does the following things:
    - generate x array if not provided
    - apply any masks given by user
    - ensure valid values in y array by masking NaN values
    - check that there are at least 'numpix_fit_thresh'
      valid pixels to fit. After NaNs are masked and SN+host
      are masked it needs these min number of pix to fit.
    """

    # Generate x array as needed
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    # Apply mask
    donotmask = cfg['donotmask']
    if mask is not None and not donotmask:
        x_fit = np.delete(x, mask)
        y_fit = np.delete(y, mask)
    else:
        x_fit = x
        y_fit = y

    # Additionally mask NaNs
    valid_mask = np.isfinite(y_fit)
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    # Also check min len requirement
    prep_flag = True
    numpix_fit_thresh = cfg['numpix_fit_thresh']
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
    except (ValueError, astropy.modeling.fitting.NonFiniteValueError) as e:
        print('\nEncountered exception:', e)
        print(x_fit)
        print(y_fit)
        print('x loc:', xloc)
        print('Row index:', row_idx)
        print('\n')

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(x_fit, y_fit, 'o', markersize=4, color='k')
        # ax.set_title('Fitting failed due to non-finite values.\n' +
        #              'Row index: ' + str(row_idx))
        # plt.show()

        # Return a NoneType model that we will check for
        # in the model gen func and skip.
        fitted_model = None
        # sys.exit(1)

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
        std = np.nanstd(arr_fit)
        return std
    else:
        return np.nanstd(arr)


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


def get_all_masks(cfg):
    # We actually need to create the masks first
    # mask the SN
    # Note that these indices are relative to the cutout indices.
    # because the cutout was already centered on xsn we can just
    # mask out hte central pixels
    snmaskpad = cfg['snmaskpad']
    galmaskpad = cfg['galmaskpad']
    sn_mask_idx = np.arange(int(cs_x/2) - snmaskpad,
                            int(cs_x/2) + snmaskpad)
    # mask the galaxy
    # print('Host cutout center idx:', cutout_host_x)
    gal_mask_idx = np.arange(cutout_host_x - galmaskpad,
                             cutout_host_x + galmaskpad)

    combinedmask = np.union1d(sn_mask_idx, gal_mask_idx)
    # print('combined mask:', combinedmask)
    return combinedmask, gal_mask_idx, sn_mask_idx


def gen_host_model(cutout, cfg):

    host_model_to_fit = 'moffat'
    sn_model_to_fit = 'moffat'

    # ----- Get user config values needed
    start_row = cfg['start_row']
    # You can change end row here for diagnostic purposes
    # The end_row should be set to the end of the cutout
    # but this can be set to the start_row + some other
    # number of rows that you'd like to see fits for
    end_row = cutout.shape[0]  # start_row + 10  # cutout.shape[0]

    # fit thresh
    sigma_thresh = cfg['sigma_thresh']
    numpix_fit_thresh = cfg['numpix_fit_thresh']

    # verbosity
    verbose = cfg['verbose']

    # ----- Masks
    combinedmask, gal_mask_idx, sn_mask_idx = get_all_masks(cfg)

    # ----- loop over all rows
    host_fit_params = {}
    host_fit_params['model'] = host_model_to_fit

    for i in range(start_row, end_row):
        # print('\nFitting row:', i)
        profile_pix = cutout[i]

        # ----- Apply a Savitsky-Golay filter to the data, if user requested
        applysmoothing = cfg['applysmoothingperrow']
        if applysmoothing:
            profile_pix = savgol_filter(profile_pix, window_length=5,
                                        polyorder=2)

        # ----- Skipping criterion
        stdrow = get_row_std_objmasked(profile_pix, mask=combinedmask)
        # print('Estimated stddev (with masked SN and host locations)',
        #       'of this row of pixels:', stdrow)
        # contiguous pixels over 3-sigma
        pix_over_thresh = (profile_pix > sigma_thresh*stdrow)  # boolean
        res = get_contiguous_slices(pix_over_thresh,
                                    min_length=numpix_fit_thresh)
        if not res:
            if verbose:
                print('Row:', i, 'Too few pixels above',
                      sigma_thresh, 'sigma to fit. Skipping.')
            continue

        # ------ Proceed to fitting
        # Fit a "modified" Sersic profile to the galaxy
        # print('\nFitting galaxy profile...')
        # print('SN mask:', sn_mask_idx)
        gal_x_fit, gal_y_fit, gal_prep_flag = prep_fit(cfg, profile_pix, xarr,
                                                       mask=sn_mask_idx)
        if not gal_prep_flag:
            continue
        galaxy_fit = fit_1d(gal_y_fit, gal_x_fit, xloc=cutout_host_x,
                            model=host_model_to_fit, row_idx=i)
        if galaxy_fit is None:
            continue

        # Fit a moffat profile to the supernova "residual"
        residual_pix = profile_pix - galaxy_fit(xarr)

        # Now mask the galaxy indices
        sn_x_fit, sn_y_fit, sn_prep_flag = prep_fit(cfg, residual_pix, xarr,
                                                    mask=gal_mask_idx)
        if not sn_prep_flag:
            continue
        moffat_fit = fit_1d(sn_y_fit, sn_x_fit,
                            model=sn_model_to_fit, row_idx=i)
        if moffat_fit is None:
            continue

        if cfg['showfit']:
            print('----------------')
            print('Galaxy fit result:')
            print(galaxy_fit)

            print('SN fit result:')
            print(moffat_fit)
            print('----------------')

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

        # Save the host fit model
        host_model_row = galaxy_fit(xarr)
        host_model[i] = host_model_row

        # Save individual fit params
        host_fit_params['Row' + str(i) + '-amp'] = galaxy_fit.amplitude.value
        # host_fit_params['Row' + str(i) + '-x0'] = galaxy_fit.x_0.value
        host_fit_params['Row' + str(i) + '-gamma'] = galaxy_fit.gamma.value
        host_fit_params['Row' + str(i) + '-alpha'] = galaxy_fit.alpha.value

    return host_model, host_fit_params


def update_host_model(cutout, hmodel, hfit_par, cfg):

    iter_flag = False

    # verbosity
    verbose = cfg['verbose']

    # get user config params needed
    start_row = cfg['start_row']

    # ---------
    # Step 1: Check fit params and smooth out host model
    """
    amplitude_arr = []
    gamma_arr = []
    alpha_arr = []
    row_idx_arr = []
    for r in range(start_row, hmodel.shape[0]):
        try:
            amplitude_arr.append(hfit_par['Row' + str(r) + '-amp'])
            gamma_arr.append(hfit_par['Row' + str(r) + '-gamma'])
            alpha_arr.append(hfit_par['Row' + str(r) + '-alpha'])
            row_idx_arr.append(r)
        except KeyError:
            continue

    row_idx_arr = np.array(row_idx_arr)
    amplitude_arr = np.array(amplitude_arr)
    gamma_arr = np.array(gamma_arr)
    alpha_arr = np.array(alpha_arr)

    # Fit a low-order polynomial to the fit params
    # First mask values where we know there isn't host light
    extrarowcutoff = 30
    valid_idx = np.where(row_idx_arr < (start_row + 210 - extrarowcutoff))[0]
    pamp, pamp_cov = np.polyfit(row_idx_arr[valid_idx],
                                amplitude_arr[valid_idx],
                                deg=1, cov=True)
    pgamma, pgamma_cov = np.polyfit(row_idx_arr[valid_idx],
                                    gamma_arr[valid_idx],
                                    deg=3, cov=True)
    palpha, palpha_cov = np.polyfit(row_idx_arr[valid_idx],
                                    alpha_arr[valid_idx],
                                    deg=3, cov=True)

    # get the error on the polynomial fit
    # these are the errors on the fit parameters
    amp_err = np.sqrt(np.diag(pamp_cov))
    gamma_err = np.sqrt(np.diag(pgamma_cov))
    alpha_err = np.sqrt(np.diag(palpha_cov))

    print(pamp)
    print(pgamma)
    print(palpha)
    print(amp_err)
    print(gamma_err)
    print(alpha_err)

    fig = plt.figure(figsize=(7, 3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    # show the best fit params
    ax1.plot(row_idx_arr, amplitude_arr, 'o', markersize=4, color='k')
    ax1.set_title(hfit_par['model'] + '-Amplitude')
    ax2.plot(row_idx_arr, gamma_arr, 'o', markersize=4, color='r')
    ax2.set_title(hfit_par['model'] + '-Gamma')
    ax3.plot(row_idx_arr, alpha_arr, 'o', markersize=4, color='b')
    ax3.set_title(hfit_par['model'] + '-Alpha')
    # show the fits
    ax1.plot(row_idx_arr, np.polyval(pamp, row_idx_arr), color='gray')
    ax2.plot(row_idx_arr, np.polyval(pgamma, row_idx_arr), color='gray')
    ax3.plot(row_idx_arr, np.polyval(palpha, row_idx_arr), color='gray')

    # also show error on fits # shaded area
    ax1.fill_between(row_idx_arr, np.polyval(pamp, row_idx_arr) - amp_err,
                     np.polyval(pamp, row_idx_arr) + amp_err,
                     color='gray', alpha=0.5)
    ax2.fill_between(row_idx_arr, np.polyval(pgamma, row_idx_arr) - gamma_err,
                     np.polyval(pgamma, row_idx_arr) + gamma_err,
                     color='gray', alpha=0.5)
    ax3.fill_between(row_idx_arr, np.polyval(palpha, row_idx_arr) - alpha_err,
                     np.polyval(palpha, row_idx_arr) + alpha_err,
                     color='gray', alpha=0.5)

    plt.show()

    sys.exit(0)
    """

    # ---------
    # Step 2: find the single rows in the host model
    # that are empty and fill them in with interpolation.
    # first find all zero rows
    zero_row_idxs = []
    for r in range(start_row, hmodel.shape[0]):
        current_row = hmodel[r]
        # test for all zeros
        if not current_row.any():
            zero_row_idxs.append(r)

    if verbose:
        print('\nFilling in single empty rows in the host model.')
        print('Empty rows:', zero_row_idxs)

    # now check for zero rows which have filled rows on either side
    rows_to_fill = []
    for i, idx in enumerate(zero_row_idxs):
        if ((zero_row_idxs[i-1] != (idx - 1))
                and (zero_row_idxs[i+1] != (idx + 1))):
            rows_to_fill.append(idx)

    for row in rows_to_fill:
        avg_row = (hmodel[row-1] + hmodel[row+1]) / 2
        hmodel[row] = avg_row

    if verbose:
        print('Filled in rows:', rows_to_fill)
        print('\n')

    # ---------
    # Step 3: Stack rows where two or more rows aren't fit
    # Find contiguous zero row idxs in the host model
    # Need to convert to boolean array first.
    # This boolean array is True where the host model has a zero row.
    zero_row_bool = np.zeros(hmodel.shape[0], dtype=bool)
    zero_row_bool[zero_row_idxs] = True
    cont_zero_rows = get_contiguous_slices(zero_row_bool, min_length=2)

    if verbose:
        print('\nStacking contiguous zero rows in the host model.')
        print('Contiguous zero row idxs. Tuples of (start, end):')
        print(cont_zero_rows)

    for j in range(len(cont_zero_rows)):
        start, end = cont_zero_rows[j]
        stack_row_idx = np.arange(start, end+1, dtype=int)

        # Skip if rows to stack aren't expected to have host light
        if start > (start_row + 210):
            continue

        if verbose:
            print('Start, end rows for stack:', start, end)

        remove_rows = np.where(stack_row_idx > (start_row + 210))[0]
        if remove_rows.size > 0:
            stack_row_idx = np.delete(stack_row_idx, remove_rows)
            if verbose:
                print('Removing rows that are not expected',
                      'to have host light.')
                print('New stack row idxs:', stack_row_idx)

        # If the number of rows to stack exceeds some number
        # over which the PSF would be expected to change a lot,
        # then we break up the stack into multiple parts.
        stack_row_idx_list = split_indices(stack_row_idx, max_length=8)
        if len(stack_row_idx_list) > 1:
            for st in stack_row_idx_list:
                galaxy_fit, stack_res = fit_stack(cutout, st)
                hmodel[st] = galaxy_fit(xarr)
        else:
            galaxy_fit, stack_res = fit_stack(cutout, stack_row_idx)
            hmodel[stack_row_idx] = galaxy_fit(xarr)

    # ---------
    # Smooth out the host model
    for col in range(hmodel.shape[1]):
        current_col = hmodel[:, col]
        newcol = savgol_filter(current_col, window_length=8, polyorder=3)
        hmodel[:, col] = newcol

    return iter_flag, hmodel, hfit_par


def fit_stack(cutout, st):
    # Mean stack
    all_rows_stack = cutout[st]
    mean_stack = np.mean(all_rows_stack, axis=0)

    # NOw fit to the stack and replace all zero rows
    # in the host model with this fit
    # We're going to force fit the mean stack regardless of the prep flag
    combinedmask, gal_mask_idx, sn_mask_idx = get_all_masks(cfg)
    gal_x_fit, gal_y_fit, gal_prep_flag = prep_fit(cfg, mean_stack, xarr,
                                                   mask=sn_mask_idx)
    gfit = fit_1d(gal_y_fit, gal_x_fit, xloc=cutout_host_x,
                  model='moffat')

    # Show stack, fit, and data that went into the stack
    if cfg['show_stack_fit']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(len(mean_stack)), mean_stack,
                   s=4, c='r', label='Stacked data')
        ax.scatter(np.arange(len(mean_stack)),
                   savgol_filter(mean_stack, window_length=4,
                                 polyorder=2),
                   s=6, c='b', label='SG smoothed data')
        ax.plot(xarr, gfit(xarr), color='g', label='Fit result')
        ax.legend(loc=0)
        plt.show()

    return gfit, mean_stack


def split_indices(indices, max_length):
    """
    Splits a continuous array of indices into
    sub-arrays of length <= max_length.

    Parameters:
        indices (array-like): Continuous array of indices.
        max_length (int): Maximum allowed length for each sub-array.

    Returns:
        list of np.ndarray: List of sub-arrays,
        each with length <= max_length.
    """
    indices = np.asarray(indices)
    n = len(indices)
    if n <= max_length:
        sub_arrays = [indices]
    else:
        sub_arrays = []
        for i in range(0, n, max_length):
            sub = indices[i:i+max_length]
            if len(sub) <= max_length:
                sub_arrays.append(sub)

    return sub_arrays


if __name__ == '__main__':

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
          'contaminated by another galaxy spectrum (only a part',
          'of the host spec will be covered by the other spectrum).')
    print('* NOTE: using the max val in the cutout as the amplitude',
          'as the initial guess. Should work in most cases.')
    print('* NOTE: SN 1D spectrum is just collapsing the 2D residuals.',
          'It calls onedutils which does the simple sum.',
          'This should use something like the Horne86 optimal extraction.')
    print('* NOTE: iterate with constraints on fit params.',
          'Also try smoothing host model before next iteration.')
    print('* NOTE: Check WCS and x,y coords returned by above func in ds9.')
    print('* NOTE: Figure out how to handle ERR and DQ extensions.')
    print('* NOTE: Show resid hist with SN masked.')
    print('* NOTE: Try grid of sims.')
    print('* NOTE: Move to testing with HST data once above items are done.')
    print('\n')

    # ==========================
    # Get configuration
    config_flname = 'user_config_1dhostsub.yaml'
    with open(config_flname, 'r') as fh:
        cfg = yaml.safe_load(fh)

    print('Received the following configuration from the user:')
    pprint(cfg)
    # ==========================

    # Read in the effective area curve for the prism
    # these areas are in square meters
    roman_effarea = np.genfromtxt(prismdir + 'Roman_effarea_20201130.csv',
                                  dtype=None, names=True, delimiter=',')
    prism_effarea = roman_effarea['SNPrism'] * 1e4  # cm2
    prism_effarea_wave = roman_effarea['Wave'] * 1e4  # angstroms

    # Read in input spectra
    # We assume that the host and SN input spectra are the same
    # for the entire file name list provided by the user.
    host_spec_fname = datadir + cfg['host_spec_fname']
    sn_spec_fname = datadir + cfg['sn_spec_fname']
    host_input_wav, host_input_flux = np.loadtxt(host_spec_fname, unpack=True)
    sn_input_wav, sn_input_flux = np.loadtxt(sn_spec_fname, unpack=True)

    fname_list = cfg['fname']
    for fname in fname_list:
        print('\nWorking on file:', fname)

        # Convert SN and host locations from user to x,y
        # Get user coords
        snra = cfg['snra']
        sndec = cfg['sndec']
        hostra = cfg['hostra']
        hostdec = cfg['hostdec']
        xsn, ysn, xhost, yhost = get_sn_host_loc(fname, snra, sndec,
                                                 hostra, hostdec)

        # Load image
        sciextnum = cfg['sciextnum']
        prismimg = fits.open(datadir + fname)
        prismdata = prismimg[sciextnum].data

        # convert to integer pixels
        row = int(ysn)
        col = int(xsn)
        # print('SN row:', row, 'col:', col)

        # get cutout size config
        cs_x = cfg['cutoutsize_x']
        cs_y_lo = cfg['cutoutsize_y_lo']
        cs_y_hi = cfg['cutoutsize_y_hi']

        # Cutout centered on SN loc
        cutout = prismdata[row - cs_y_lo: row + cs_y_hi,
                           col - int(cs_x/2): col + int(cs_x/2)]

        # Get host location within cutout
        cutout_host_x = int(cs_x/2) + int(int(xhost) - col)

        # plot
        if cfg['showcutoutplot']:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(np.log10(cutout), origin='lower',
                            vmin=1.5, vmax=2.5)
            cbar = fig.colorbar(cax)
            cbar.set_label('log(pix val)')
            plt.show()
            fig.savefig(fname.replace('.fits', '_cutout.png'), dpi=200,
                        bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # ==========================
        # Now fit the profile
        xarr = np.arange(cs_x)

        # Empty array for host model
        host_model = np.zeros_like(cutout)
        # Fit and update
        host_model, hf_par = gen_host_model(cutout, cfg)
        iter_flag, host_model, host_fit_params = update_host_model(cutout,
                                                                   host_model,
                                                                   hf_par,
                                                                   cfg)

        # Iterate and test
        """
        num_iter = 1
        max_iter = cfg['max_iter']
        while num_iter < max_iter:
            print('Iteration:', num_iter)
            host_model, host_fit_params = gen_host_model(cutout, cfg)
            iter_flag, host_model = update_host_model(cutout, host_model,
                                                      host_fit_params, cfg)
            if iter_flag:
                num_iter += 1
                continue
            else:
                break
        print('\nDone in', num_iter, 'iterations.\n')
        """

        # ==========================
        # Get the SN spectrum
        recovered_sn_2d = cutout - host_model

        imghdr = prismimg[sciextnum].header
        wcs = WCS(imghdr)
        # WCS unused if coordtype is 'pix'.
        # The object X, Y are the center of the cutout
        # and the 1.55 micron row index.
        obj_x = int(cs_x/2)
        obj_y = cutout.shape[0] - cs_y_hi
        # print(obj_x, obj_y)
        # Extraction params from config
        one_sided_width = cfg['obj_one_sided_width']
        swav = cfg['start_wav']
        ewav = cfg['end_wav']
        rs, re, cs, ce, specwav = oned_utils.get_bbox_rowcol(obj_x, obj_y, wcs,
                                                             one_sided_width,
                                                             coordtype='pix',
                                                             start_wav=swav,
                                                             end_wav=ewav)
        # print('\n1D-Extraction params:')
        # print('Row start:', rs, 'Row end:', re)
        # print('Col start:', cs, 'Col end:', ce)
        # print('Wavelengths:', specwav)
        # print(specwav.shape)

        # Collapse to 1D
        # Since we know the location of the SN, we will just
        # use a few pixels around it.
        spec2d = recovered_sn_2d[rs: re + 1, cs: ce + 1]
        sn_1d_spec = np.nanmean(spec2d, axis=1)

        # convert to physical units
        et = cfg['spec_img_exptime']
        sn_1d_spec_phys = oned_utils.convert_to_phys_units(spec2d, specwav,
                                                           'DN',
                                                           prism_effarea_wave,
                                                           prism_effarea,
                                                           subtract_bkg=False,
                                                           spec_img_exptime=et)

        # also try showing what you'd get if you didn't subtract the host
        spec2d_with_host = cutout[rs: re + 1, cs: ce + 1]
        sn_1d_phys_host = oned_utils.convert_to_phys_units(spec2d_with_host,
                                                           specwav, 'DN',
                                                           prism_effarea_wave,
                                                           prism_effarea,
                                                           subtract_bkg=False,
                                                           spec_img_exptime=et)

        # ==========
        # Show all host subtraction
        fig = plt.figure(figsize=(6, 6))

        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.imshow(np.log10(cutout), origin='lower', vmin=1.5, vmax=2.5)
        ax1.set_title('SN + Host original image cutout')

        ax2.imshow(np.log10(host_model), origin='lower', vmin=1.5, vmax=2.5)
        ax2.set_title('File: ' + fname + '\n' + 'Host model')

        ax3.imshow(np.log10(recovered_sn_2d), origin='lower',
                   vmin=1.5, vmax=2.5)
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

        host_smaller_cutout = host_model[:,
                                         cutout_host_x - 25:cutout_host_x + 25]
        host_rec_flux = np.mean(host_smaller_cutout, axis=1)

        sn_input_wav_microns = sn_input_wav/1e4

        ax1.plot(specwav, sn_1d_spec_phys, '-',
                 color='crimson', lw=1.5, label='Recovered SN spec [flam]')
        ax1.plot(sn_input_wav_microns, sn_input_flux, '-',
                 color='mediumseagreen', label='Input SN spec')

        # Also plot the SN spectrum without host contamination subtracted
        ax1.plot(specwav, sn_1d_phys_host, '-',
                 color='slategray', lw=1.5,
                 label='SN spec without host\n' + 'contam. subtracted')

        ax1.legend(loc=0, fontsize=10)

        ax1.set_xlim(0.65, 2.0)
        ax1.set_ylim(0, 1.5e-19)

        ax1.set_xticklabels([])

        # plot residuals
        sn_input_flux_inwav = griddata(points=sn_input_wav_microns,
                                       values=sn_input_flux,
                                       xi=specwav)
        spec_resid = ((sn_input_flux_inwav - sn_1d_spec_phys)
                      / sn_input_flux_inwav)

        ax2.scatter(specwav, spec_resid, s=4, c='k')

        ax2.set_xlim(0.65, 2.0)
        ax2.set_ylim(-1.5, 1.5)

        fig.savefig(fname.replace('.fits', '_1dparamfit_phys.png'),
                    dpi=200, bbox_inches='tight')
        # plt.show()

        # Close image
        prismimg.close()

    sys.exit(0)
