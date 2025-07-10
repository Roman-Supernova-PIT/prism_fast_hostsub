#from astropy.convolution import convolve
import numpy as np
from scipy.linalg import cholesky
    

def boxsmooth(arr, widx, widy):
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
    sx0 = 8
    sy0 = 8

    px0 = 1
    py0 = 1
    
    tilex = 2
    tiley = 2
    
    dx = (widx-1)//2
    dy = (widy-1)//2

    padx = Np+dx+px0
    pady = Np+dy+px0
    
    
    stepx = (sx0+2)//tilex
    stepy = (sy0+2)//tiley
    
    
    
    bism = np.zeros((stepx+2*padx-2*dx,stepy+2*pady-2*dy,2*Np-1, Np),
        dtype=float)
    sy1, sx1 = arr.shape
    
    
    bimage, tot = boxsmooth(arr, widx, widy)
    
    for dc in range(Np):
        for dr in range(1-Np, Np):
            if dr < 0 and dc == 0:
                pass
            else:

                #ism = arr*np.roll(arr, (-dr, -dc))
                ism = arr*np.roll(np.roll(arr, -dc, axis=0), -dr, axis=1)
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
                pass
            else:
                if dr>= 0:
                    prr = range(Np-dr)
                if dr < 0 and dc >0:
                    #prr = range(1-dr-1, Np)
                    prr = range(-dr, Np)

                for pc in pcr:
                    for pr in prr:
                        
                        i = (pc)*Np+pr
                        j = ((pc+dc)*Np)+pr+dr
                    
                        a1a2 = bimage[pc+delc, pr+delr]*bimage[pc+dc+delc, pr+dr+delr]/(widx*widy)**2
                        t = bism[pc+delc, pr+delr, dr+Np-1, dc]/(widx*widy) - a1a2

                        cov[i, j] = cov[j, i] = t
        
                        if i == j:
                            ave[i] = a1a2

    cov *= (widx*widy)/((widx*widy)-1)
    return cov, ave

def _extract_submatrix(matrix, i, j):
    n = np.count_nonzero(i)
    m = np.count_nonzero(j)
    mat = matrix[np.where(np.outer(i, j))].reshape(n, m)
    return mat

def _symmetrize_matrix(matrix):
    m = matrix+matrix.T-np.diag(matrix.diagonal())
    return m
    
def condCovEst_wdiag(cov_loc, ave, km, kpsf2d, data, stars, psft, Np=3, export_mean=True, n_draw=2, diag_on=True):
    k = np.logical_not(km)
    kstar = kpsf2d.flatten()
    # kstardim = kpsf2d.shape()

    if diag_on:
        s = stars.flatten()
        for i in range(Np*Np):
            cov_loc[i,i] += s[i]
    cov_kk = _symmetrize_matrix(_extract_submatrix(cov_loc, k, k))
    cov_kkstar = _extract_submatrix(cov_loc, k, kstar)
    cov_kstarkstar = _extract_submatrix(cov_loc, kstar, kstar)

    #C = cholesky(cov_kk)
    I = np.linalg.inv(cov_kk)

    icovkkCcovkkstar = np.dot(I, cov_kkstar)

    
    predcovar = _symmetrize_matrix(cov_kstarkstar-np.dot(cov_kkstar.T,icovkkCcovkkstar))

    uncond = data.T.flatten()
    cond = uncond-ave

    kstarpredn = np.dot(cond[k].T, icovkkCcovkkstar)
    kstarpred = kstarpredn + ave[kstar]

    p = psft[kpsf2d].flatten()

    ipcovCp =  np.dot(np.linalg.inv(predcovar), p)

    var_wdb = np.dot(p.T, ipcovCp)

    
    var_diag = 0
    for i in range(predcovar.shape[1]):
        var_diag += (p[i]*p[i])/predcovar[i,i]

        
    resid = np.dot(uncond[kstar], ipcovCp)/var_wdb
    pred = np.dot(kstarpred, ipcovCp)/var_wdb

    chi2 = np.dot(cond[k], np.dot(I, cond[k]))
    out = [1./np.sqrt(var_wdb), 1./np.sqrt(var_diag), pred-resid, resid, chi2]

    if export_mean:
        mean = data.copy()
        mean[np.where(kpsf2d.T)] = kstarpred
        out.append(mean)

    if n_draw > 0:
        U, S, Vh = np.linalg.svd(predcovar, full_matrices=False)
        sqrt_cov = np.dot(Vh, np.dot(np.diag(np.sqrt(S)), Vh.T))
        noise = np.dot(sqrt_cov, np.random.normal(size=(sqrt_cov.shape[0], n_draw)))

        d = data.T.flatten()
        draw_out = np.empty((len(d), n_draw), dtype=data.dtype)
        for i in range(n_draw):
            draw_out[:, i] = d

        j = 0
        for i in range(len(d)):
            
            if kstar[i]:
                draw_out[i,:] = noise[:, j]
                j += 1
        out.append(draw_out)
    return out

if __name__ == '__main__':
    x = np.arange(1, 122).reshape(11, 11).T
    #out, tot = boxsmooth(x, 3, 3)

    #kern = np.ones((3, 3), dtype=float)/9
    #q = convolve(x, kern).astype(int)
    #print(q)

    Np = 5
    widx = 3
    widy = 3
    
    x = np.arange(1, 19*19+1).reshape(19, 19).T
    #bimage, bism, ism = cov_avg(x, Np=Np, widx=widx, widy=widy)


    Np = 5
    widx = 3
    widy = 3
    px0 = 1
    py0 = 1
    sx0 = 8
    sy0 = 8
    tilex = 2
    tiley = 2
    
    dx = (widx-1)//2
    dy = (widy-1)//2
    padx = Np+dx+px0
    pady = Np+dy+py0
    stepx = (sx0+2)//tilex
    stepy = (sy0+2)//tiley

    #x = np.arange(1, (stepx+2*padx)*(stepy+2*pady)+1).reshape(stepy+2*pady,stepx+2*padx)

    #bimage, bism, ism = cov_avg(x, Np=Np, widx=widx, widy=widy)

    #cov, ave = build_cov(6, 6, bimage, bism, Np, widx, widy)
    

    kpsf2d = np.zeros((3,3), dtype=bool)
    kpsf2d[1,1] = True
    kpsf2d[2,1] = True

    
    km = kpsf2d.copy()
    km = km.reshape(3*3)
    km[0] = True
    km[-1] = True

    data = np.array([[0.516654, 0.128205, 1.94308],
        [-0.0805772, -0.920908, 0.212453],
        [-0.774471, 0.165229, -0.363535]])

    stars = np.full((3,3), 2, dtype=int)
    cov_loc = np.array([[47.2245, 1.33946, 0.628186, 0.841306, 0.288469, -0.437706, 0.754434, -0.245601, -0.150857],
        [0.0, 47.2602, 1.39354, 0.570245, 0.871201, 0.243023, -0.457453, 0.806491, -0.209147],
        [0.0, 0.0, 47.166, 1.40991, 0.471935, 0.767516, 0.251835, -0.398556, 0.772153],
        [0.0, 0.0, 0.0, 47.1832, 1.44052, 0.461676, 0.757649, 0.282371, -0.314066],
        [0.0, 0.0, 0.0, 0.0, 47.2298,1.34296, 0.471258, 0.766785, 0.218381],
        [0.0, 0.0, 0.0, 0.0, 0.0, 47.2845, 1.36993, 0.492573, 0.715498],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.1459, 1.3015, 0.425096],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.1451, 1.30432],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47.2068]])

    ave = np.array([0.005037597380578518, 0.003629490500316024, 0.0026571564376354218, 0.0032599247060716152, 0.002772299339994788, 0.0034678715746849775, 0.0051109688356518745, 0.0037797277327626944, 0.003628455102443695])
    psft = np.array([[0.0274769, 0.0360258, 0.032605],
        [0.0330318, 0.0403823, 0.0339776],
        [0.0301219, 0.033388, 0.0255433]])

    out = condCovEst_wdiag(cov_loc, ave, km, kpsf2d, data, stars, psft, Np=3, export_mean=False, n_draw=0, diag_on=True)    out = condCovEst_wdiag(cov_loc, ave, km, kpsf2d, data, stars, psft, Np=3, export_mean=True, n_draw=0, diag_on=True)
    out = condCovEst_wdiag(cov_loc, ave, km, kpsf2d, data, stars, psft, Np=3, export_mean=True, n_draw=2, diag_on=True)

 
    

    
