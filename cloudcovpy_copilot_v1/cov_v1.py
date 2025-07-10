import numpy as np
from scipy.linalg import cholesky


def outest_bounds(cx, sx):
    """
    Helper function to find maximum padding in pixels required to
    accommodate all query points cx outside of the image size 1:sx.
    """
    px0 = 0
    sortcx = np.sort(cx)
    if sortcx[0] < 1:
        px0 = abs(sortcx[0] - 1)
    if sortcx[-1] > sx:
        if px0 < (sortcx[-1] - sx):
            px0 = (sortcx[-1] - sx)
    return px0


def boxsmooth(arr, widx, widy):
    """
    Boxcar smooths an input image arr with window size widx by widy.
    """

    sy, sx = arr.shape  # row, col

    tot = np.zeros(sx, dtype=float)
    out = np.zeros((sy - widy + 1, sx - widx + 1), dtype=float)

    for j in range(sy - widy + 1):
        if j == 0:
            for n in range(widy):
                tot += arr[:, n]
        else:
            tot += arr[:, j + widy - 1] - arr[:, j - 1]
        tt = 0
        for i in range(sx - widx + 1):
            if i == 0:
                tt = np.sum(tot[:widx])
            else:
                tt += tot[i + widx - 1] - tot[i - 1]
            out[i, j] = tt

    return out


def cov_avg(in_image, Np=33, widx=129, widy=129):
    """
    Constructs shifted and multiplied versions of the
    input image for local covariance estimation.
    """

    sx1, sy1 = in_image.shape

    bimage = boxsmooth(in_image, widx, widy)

    for dc in range(Np):
        for dr in range(1 - Np, Np):
            if dr < 0 and dc == 0:
                continue
            ism[:] = in_image * np.roll(in_image, shift=(-dr, -dc), axis=(0, 1))

            bism[:, :, dr + Np, dc] = boxsmooth(ism, widx, widy)

    return bimage, bism, ism


def prelim_infill(testim, bmaskim, testim2, bmaskim2, goodpix,
                  widx=19, widy=19, widmult=1.4):
    """
    Initial infill replaces masked pixels with a
    guess based on a smoothed boxcar.
    """

    sx, sy = testim.shape

    widx_max = int(round((widmult**10)*(widx-1)/2))*2+1
    widy_max = int(round((widmult**10)*(widy-1)/2))*2+1
    half_widx_max = (widx_max-1)//2
    half_widy_max = (widy_max-1)//2

    # Set masked entries in testim to 0
    testim = testim.copy()
    testim[bmaskim] = 0
    np.copyto(bmaskim2, bmaskim)
    np.copyto(testim2, testim)

    # Pad arrays with reflection
    in_image = np.pad(testim, ((half_widx_max, half_widx_max), (half_widy_max, half_widy_max)), mode='reflect')
    in_mask = np.pad(~bmaskim, ((half_widx_max, half_widx_max), (half_widy_max, half_widy_max)), mode='reflect')

    cnt = 0
    while np.any(bmaskim2) and cnt < 10:
        half_widx = (widx-1)//2
        half_widy = (widy-1)//2
        # Slicing to match Julia's 1-based indexing
        in_image1 = in_image[half_widx:half_widx+sx, half_widy:half_widy+sy]
        in_mask1 = in_mask[half_widx:half_widx+sx, half_widy:half_widy+sy]

        bimage = boxsmooth(in_image1, widx, widy)
        bimageI = boxsmooth(in_mask1, widx, widy)

        np.copyto(goodpix, bimageI > 10)

        update_mask = bmaskim2 & goodpix
        testim2[update_mask] = (bimage / bimageI)[update_mask]
        bmaskim2[goodpix] = False

        cnt += 1
        widx = int(round((widx * widmult - 1) / 2)) * 2 + 1
        widy = int(round((widy * widmult - 1) / 2)) * 2 + 1

    print(f"Infilling completed after {cnt} rounds",
          f"with final width (widx,widy) = ({widx},{widy})")

    # Catastrophic failure fallback
    if cnt == 10:
        testim2[bmaskim2] = np.median(testim)
        print("Infilling Failed Badly")

    return None


if __name__ == "__main__":

    # Example image (random data)
    sx, sy = 256, 256  # image size
    in_image = np.random.rand(sx, sy).astype(float)

    # Parameters
    Np = 33
    widx = 15
    widy = 15

    # Preallocate arrays
    bimage = np.zeros((sx - widx + 1, sy - widy + 1), dtype=float)
    ism = np.zeros_like(in_image)
    bism_shape = (sx - widx + 1, sy - widy + 1, 2*Np, Np)
    bism = np.zeros(bism_shape, dtype=float)

    # Example: outest_bounds
    cx = np.array([10, 20, 300])  # example star centers
    sx_dim = 256
    padding = outest_bounds(cx, sx_dim)
    print("Padding needed:", padding)

    # Example: boxsmooth
    bimage = boxsmooth(in_image, widx, widy)
    print("Boxsmoothed image shape:", bimage.shape)

    # Example: cov_avg
    bimage, ism, bism = cov_avg(in_image, Np=Np, widx=widx, widy=widy)
    print("Covariance average computed.")
