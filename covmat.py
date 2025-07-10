import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models


import sys


def outest_bounds(cx, sx):
    """
    Helper function to find maximum padding in pixels required to
    accomodate all query points `cx` outside of the image size 1:`sx`.

    # Arguments:
    - `cx`: list of integer star centers (in either x or y)
    - `sx`: image dimension along the axis indexed by `cx`

    # Outputs:
    - `px0`: maximum padding in pixels required to accomodate all query points
    """

    px0 = 0
    sort_cx = np.sort(cx)

    if sort_cx[0] < 1:
        px0 = np.abs(sort_cx[0] - 1)

    if sort_cx[-1] > sx:
        if px0 < np.abs(sort_cx[-1] - sx):
            px0 = np.abs(sort_cx[-1] - sx)

    return px0


def boxsmooth(arr, widx, widy):
    """
    Boxcar smooths an input image (or paddedview) `arr` with
    window size `widx` by `widy`. We pass the original image
    size `sx` and `sy` to help handle image views.

    The julia code is below which is converted to python.

    function boxsmooth!(out::AbstractArray, arr::AbstractArray,
    tot::Array{T,1}, widx::Int, widy::Int) where T
        (sx, sy) = size(arr)

        for j=1:(sy-widy+1)
            if (j==1)
                for n = 1:widy
                    @simd for m = 1:sx
                        @inbounds tot[m] += arr[m,n]
                    end
                end
            else
                @simd for m = 1:sx
                    @inbounds tot[m] += arr[m,j+widy-1]-arr[m,j-1]
                end
            end
            tt=zero(eltype(out))
            for i=1:(sx-widx+1)
                if (i==1)
                    @simd for n=1:widx
                        @inbounds tt += tot[n]
                    end
                else
                    @inbounds tt += tot[i+widx-1]-tot[i-1]
                end
                @inbounds out[i,j] = tt
            end
        end
    end
    """

    sy, sx = arr.shape
    dx = (widx-1)//2
    dy = (widy-1)//2

    tot = np.zeros(sx, dtype=int)
    out = np.zeros((sy-2*dy, sx-2*dx), dtype=int)

    for j in range(sy-widy+1):
        if j == 0:
            for n in range(widy):
                for m in range(sx):
                    tot[m] += arr[n, m]
        else:
            for m in range(sx):
                tot[m] += (arr[j+widy-1, m] - arr[j-1, m])

        tt = 0
        for i in range(sx-widx+1):
            if i == 0:
                for n in range(widx):
                    tt += tot[n]

            else:
                tt += tot[i+widx-1]-tot[i-1]

            out[j, i] = tt

    return out, tot


def cov_avg(arr, Np=33, widx=129, widy=129):
    """
    # =======
    Julia docstring and function below:

        cov_avg!(bimage, ism, bism, in_image; Np::Int=33,
        widx::Int=129, widy::Int=129, ftype::Int=32)

    Key function for constructing the (shifted and multiplied
    versions of the input image used to quickly
    estimate the local covariance matrix at a large number
    of locations. The main output is in the preallocated
    `bism` which is used as an input to `build_cov!`.

    # Arguments:
    - `bimage`: preallocated output array for the
                boxcar smoothed unshifted image
    - `ism`: preallocated intermediate array for the
             input image times itself shifted
    - `bism`: preallocated output array to store
              boxcar-smoothed image products for all shifts
    - `in_image`: input image the local covariance
                  of which we want to estimate

    # Keywords:
    - `Np::Int`: size of local covariance matrix
                 in pixels (default 33)
    - `widx::Int`: width of boxcar window in x which
                   determines size of region used for
                   samples for the local covariance
                   estimate (default 129)
    - `widy::Int`: width of boxcar window in y which
                   determines size of region used for
                   samples for the local covariance
                   estimate (default 129)
    - `ftype::Int`: determine the Float precision,
                    32 is Float32, otherwise Float64

    function cov_avg!(bimage, ism, bism, in_image;
    Np::Int=33, widx::Int=129, widy::Int=129, ftype::Int=32)
        if ftype == 32
            T = Float32
        else
            T = Float64
        end

        (sx1, sy1) = size(in_image)
        tot = zeros(T,sx1);
        boxsmooth!(bimage,in_image,tot,widx,widy)
        # loop over shifts
        for dc=0:Np-1       # column shift loop
            for dr=1-Np:Np-1   # row loop, incl negatives
                if (dr < 0) & (dc == 0)
                    continue
                end
                # ism = image, shifted and multipled
                @inbounds ism .= in_image .*
                ShiftedArrays.circshift(in_image,(-dr, -dc))
                fill!(tot,0)
                boxsmooth!(view(bism,:,:,dr+Np,dc+1),ism,tot,widx,widy)
                # bism = boxcar(ism)
            end
        end
        return
    end
    """

    sy1, sx1 = arr.shape

    # Get the smoothed image first
    bimage, tot = boxsmooth(arr, widx, widy)

    # Construct the empty array to hold shifted and multiplied images
    dx = (widx-1)//2
    dy = (widy-1)//2
    bism = np.zeros((sy1-2*dy, sx1-2*dx, 2*Np-1, Np), dtype=float)

    # loop over shifts
    for dc in range(Np):  # column shift loop
        for dr in range(1-Np, Np):  # row loop, incl negatives
            if (dr < 0) and (dc == 0):
                continue

            ism = arr * np.roll(np.roll(arr, -dc, axis=0), -dr, axis=1)
            b, tot = boxsmooth(ism, widx, widy)
            bism[:, :, dr+Np-1, dc] = b

    return bimage, bism, ism


def build_cov(cx, cy, bimage, bism, Np, widx, widy):

    halfNp = (Np-1)//2
    delr = cx-(halfNp+1)
    delc = cy-(halfNp+1)

    ave = np.zeros(Np*Np, dtype=float)
    cov = np.zeros((Np*Np, Np*Np), dtype=float)

    for dc in range(Np):
        pcr = range(Np-dc)
        for dr in range(1-Np, Np):

            if dr < 0 and dc == 0:
                continue
            else:
                if dr >= 0:
                    prr = range(Np-dr)
                if dr < 0 and dc > 0:
                    prr = range(-dr, Np)

                for pc in pcr:
                    for pr in prr:

                        i = pc*Np + pr
                        j = ((pc+dc)*Np) + pr + dr

                        a1a2 = ((bimage[pc+delc, pr+delr]
                                 * bimage[pc+dc+delc, pr+dr+delr])
                                / (widx*widy)**2)
                        t = (bism[pc+delc, pr+delr, dr+Np-1, dc]/(widx*widy)
                             - a1a2)

                        cov[i, j] = cov[j, i] = t

                        if i == j:
                            ave[i] = a1a2

    cov *= (widx*widy)/((widx*widy)-1)

    return cov, ave


if __name__ == "__main__":
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

    # Smoothing
    widx = 3
    widy = 3
    smoothed_arr, total = boxsmooth(arr, widx, widy)

    print("Original Array Shape:", arr.shape)
    print("Smoothed Array Shape:", smoothed_arr.shape)
    print("Total Array Shape:", total.shape)
    print("Padding required:",
          outest_bounds([1, 2, 3, 4, 5], arr.shape[1]))

    # Plot the original and smoothed arrays side by side
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(arr, cmap='gray_r', origin='lower')
    ax1.set_title('Original Image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(smoothed_arr, cmap='gray_r', origin='lower')
    ax2.set_title('Boxcar Smoothed Image')
    # plt.show()

    bimage, bism, ism = cov_avg(arr, Np=20, widx=5, widy=5)

    sys.exit(0)
