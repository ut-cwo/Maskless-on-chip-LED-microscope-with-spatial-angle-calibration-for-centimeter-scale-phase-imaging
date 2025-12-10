from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift
import jax
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import jax.numpy as jnp
from timeit import default_timer as timer
from random import sample
from jax.lax import dynamic_slice, reduce_window, add
from PIL import Image
from functools import partial
import scipy
from scipy.optimize import minimize
import scipy.io
from scipy.ndimage import distance_transform_edt
from jax import jit, lax
import glob
from scipy.io import savemat
import os
from jax.numpy import pi

# Original Windows path (for reference only):
# D:\Manuscripts\Lensless_2D\Dataset\QPI_phase_target

# In WSL, Windows drives are mounted under /mnt/<drive_letter>/
# So D:\... becomes /mnt/d/...
wsl_base = '/mnt/d/Manuscripts/Lensless_2D/Dataset/QPI_phase_target/'

# Full path to input .mat file and output .mat file
mat_file = os.path.join(wsl_base, 'FOV01', 'patch_1.mat')
filename = os.path.join(wsl_base, 'Recon_python.mat')

# Load raw measurements + metadata from MATLAB file
data = scipy.io.loadmat(mat_file)

# Print keys once to verify what variables exist in the .mat file
for key in data.keys():
    print(key)

# Measurement stack: originally (rows, cols, angles), rearranged to (angles, rows, cols)
A_acqs  = data["A_acqs"]
A_acqs  = jnp.swapaxes(A_acqs, 0, 2)
A_acqs  = jnp.swapaxes(A_acqs, 1, 2)

# Scalar parameters (convert from 0-dim arrays to Python scalars)
lambdaa = data["lambda"].item()   # wavelength
dpix    = data["dpix"].item()     # sensor pixel size
sig     = data["sig"].item()      # downsampling / binning factor
do      = data["do"].item()       # object-to-sensor distance
pdar    = data["pdar"].item()     # padding size

# Corrected shifts and angles (C = corrected)
Shift_x_C      = jnp.squeeze(data["Shift_x_C"])
Shift_y_C      = jnp.squeeze(data["Shift_y_C"])

thetax_illum_C = jnp.squeeze(data["thetax_illum_C"])
thetay_illum_C = jnp.squeeze(data["thetay_illum_C"])

fx_illum_C     = jnp.squeeze(data["fx_illum_C"])
fy_illum_C     = jnp.squeeze(data["fy_illum_C"])

# Uncorrected shifts and angles (original)
Shift_x        = jnp.squeeze(data["Shift_x"])
Shift_y        = jnp.squeeze(data["Shift_y"])

thetax_illum   = jnp.squeeze(data["thetax_illum"])
thetay_illum   = jnp.squeeze(data["thetay_illum"])

fx_illum       = jnp.squeeze(data["fx_illum"])
fy_illum       = jnp.squeeze(data["fy_illum"])

# Flags and iteration controls
ang_corr      = 1      # enable illumination-angle correction
C_use         = 1      # use corrected (C) calibration if 1, raw if 0
irer_num      = 45     # number of initial phase-only iterations
shift_max     = 0.1    # max allowed shift update (pixels)
iter_Start    = 10     # start angle/shift correction at this iter
iter_Stop     = 55     # stop angle/shift correction at this iter

# Reconstruction and regularization parameters
batch_size    = 5
N_iter        = 100
step_size     = 2
regparam      = 2e-3

# Spectrum / Wiener filter params
sigma         = 220000
W_cut         = 0.8
Kotfval       = 0.2
Apo_cut       = 1.3
Index         = 0.2

# Sub-pixel registration parameters
usfac         = int(400)                # upsampling factor
no            = int(jnp.ceil(usfac*1.5))
t_k           = 1                       # FISTA momentum variable
NA            = 1                       # numerical aperture cutoff

# Basic sizes and arrays
N_illum   = A_acqs.shape[0]             # number of illumination angles
n_batches = int(jnp.ceil(N_illum / batch_size))
m         = A_acqs.shape[1]             # low-res image size (per dimension)
cost      = np.zeros(N_iter)           # cost function history
F_illum   = jnp.zeros((N_illum, 2))     # current (fx, fy) for each LED

n         = m * sig                     # high-res object size (no padding)
ps        = dpix / sig                  # effective pixel size at object plane
N         = int(n + 2 * pdar)           # padded size

# Spatial and frequency grids
x      = ps * jnp.arange(-N/2, N/2)
[X, Y] = jnp.meshgrid(x, -x)

dfx    = 1 / (N * ps)
fx     = dfx * jnp.arange(-N/2, N/2)
[Fx, Fy] = jnp.meshgrid(fx, -fx)
Fx     = ifftshift(Fx)
Fy     = ifftshift(Fy)

# Choose whether to use corrected or uncorrected illumination frequencies/angles
if C_use == 0:
    fx_illum     = (np.fix(fx_illum / dfx) * dfx).squeeze()
    fy_illum     = (np.fix(fy_illum / dfx) * dfx).squeeze()
    # Keep original Shift_x, Shift_y, thetax_illum, thetay_illum
else:
    fx_illum     = (np.fix(fx_illum_C / dfx) * dfx).squeeze()
    fy_illum     = (np.fix(fy_illum_C / dfx) * dfx).squeeze()
    Shift_x      = Shift_x_C
    Shift_y      = Shift_y_C
    thetax_illum = thetax_illum_C
    thetay_illum = thetay_illum_C

# Store initial illumination frequencies into F_illum matrix
F_illum = F_illum.at[:, 0].set(fx_illum)
F_illum = F_illum.at[:, 1].set(fy_illum)

# Backup pre-correction illumination and shifts (for plotting "before vs after")
fx_illum_b     = fx_illum
fy_illum_b     = fy_illum
Shift_x_b      = Shift_x
Shift_y_b      = Shift_y
thetax_illum_b = jnp.arcsin(-fx_illum * lambdaa)
thetay_illum_b = jnp.arcsin(-fy_illum * lambdaa)


def FTpad(imFT):
    Nin          = imFT.shape
    Nout         = tuple(2*x for x in Nin)
    center       = jnp.floor(jnp.array(Nin)/2)
    centerout    = jnp.floor(jnp.array(Nout)/2)
    cenout_cen   = centerout - center
    imFT         = fftshift(imFT)
    imFTout      = lax.full(tuple(Nout), 0, dtype=imFT.dtype)
    start_output = jnp.maximum(cenout_cen, 0).astype(int)
    imFTout1     = lax.dynamic_update_slice(imFTout, imFT, start_output)
    imFTout2     = ifftshift(imFTout1) * (Nout[0] * Nout[1]) / (Nin[0] * Nin[1])
    return imFTout2

@partial(jax.vmap, in_axes=(0, 0, None, None))
def dftregistration(Ashift, Aref,usfac,no):
    buf1ft              = fft2(Ashift**2)
    buf2ft              = fft2(Aref**2)  
    nr                  = Ashift.shape[0]
    nc                  = Ashift.shape[1]
    dftshift            = jnp.fix(jnp.ceil(usfac*1.5)/2)
    CC                  = ifft2(jit_FTpad(buf1ft * jnp.conj(buf2ft)))
    CCabs               = jnp.abs(CC)
    max_index           = jnp.argmax(CCabs)
    row_shift,col_shift = jnp.unravel_index(max_index, CCabs.shape)
    CCmax               = CC[row_shift,col_shift]*nr*nc        
    Nr                  = ifftshift(lax.iota(jnp.int32, 2*nr) - nr)
    Nc                  = ifftshift(lax.iota(jnp.int32, 2*nc) - nc)
    row_shift           = Nr[row_shift] / 2
    col_shift           = Nc[col_shift] / 2
    A                   = (ifftshift(lax.iota(jnp.int32,nc))-jnp.floor(nc/2)).reshape(-1, 1)
    B                   = lax.iota(jnp.int32,no)
    C                   = B.reshape(-1, 1)
    D                   = ifftshift(lax.iota(jnp.int32,nr)) - jnp.floor(nr/2)
    E                   = -1j*2*jnp.pi/(nc*usfac)    
    row_shift           = jnp.round(row_shift*usfac)/usfac
    col_shift           = jnp.round(col_shift*usfac)/usfac    
    roff                = dftshift - row_shift * usfac
    coff                = dftshift - col_shift * usfac 
    In                  = buf2ft*jnp.conj(buf1ft)
    kernc               = jnp.exp(E*A*(B-coff))
    kernr               = jnp.exp(E*(C-roff)*D)
    CC                  = jnp.conj(kernr@In@kernc)
    max_index           = jnp.argmax(jnp.abs(CC))
    rloc,cloc           = jnp.unravel_index(max_index, CC.shape)
    CCmax               = CC[rloc, cloc]   
    rloc                = rloc - dftshift
    cloc                = cloc - dftshift
    rs                  = row_shift + rloc / usfac
    cs                  = col_shift + cloc / usfac       
    return jnp.array([rs,cs])

def updateFill(shift_opt,Shift_x_err,Shift_y_err):
    shift_diff_x =  shift_opt[:, 1]
    shift_diff_y = -shift_opt[:, 0]  
    shift_diff_x = jnp.where(jnp.abs(shift_diff_x) >= jnp.abs(shift_max),jnp.sign(shift_diff_x)*shift_max,shift_diff_x)
    shift_diff_y = jnp.where(jnp.abs(shift_diff_y) >= jnp.abs(shift_max),jnp.sign(shift_diff_y)*shift_max,shift_diff_y)
    si_x         = Shift_x_err-shift_diff_x
    si_y         = Shift_y_err-shift_diff_y
    F_illum_x    = jnp.fix((1/lambdaa)*(jnp.sin(jnp.atan2(si_x*dpix,do)))/dfx)*dfx
    F_illum_y    = jnp.fix((1/lambdaa)*(jnp.sin(jnp.atan2(si_y*dpix,do)))/dfx)*dfx
    return F_illum_x,F_illum_y,si_x,si_y

def gradient_op2d(I, wx=1, wy=1):
    XX = I[1:, :] - I[:-1, :]
    TT = jnp.zeros((1, I.shape[1]))
    dx = jnp.concatenate((XX, TT), 0)
    YY = I[:, 1:] - I[:, :-1]
    TT = jnp.zeros((I.shape[0], 1))
    dy = jnp.concatenate((YY, TT), 1)
    return dx * wx, dy * wy

def div_op2d(dx, dy, wx=1, wy=1):
    dx = dx * jnp.conj(wx)
    dy = dy * jnp.conj(wy)
    T1 = jnp.expand_dims(dx[0, :], axis=0)
    T2 = dx[1:-1, :] - dx[:-2, :]
    T3 = jnp.expand_dims(-dx[-2, :], axis=0)
    I1 = jnp.concatenate((T1, T2, T3), 0)
    T1 = jnp.expand_dims(dy[:, 0], axis=1)
    T2 = dy[:, 1:-1] - dy[:, 0:-2]
    T3 = jnp.expand_dims(-dy[:, -2], axis=1)
    I2 = jnp.concatenate((T1, T2, T3), 1)
    I  = I1 + I2
    return I

def norm_tv2d(u, wx=1, wy=1, wz=1):
    [dx, dy] = gradient_op2d(u, wx, wy)
    temp     = jnp.sqrt(abs(dx) ** 2 + abs(dy) ** 2)
    y        = temp.sum()
    return y

@jax.jit
def prox_tv2d_step(x, gamma, prev_obj, r, s, told, pold, qold, wx, wy, mt, tol):
    sol      = x - gamma * div_op2d(r, s, wx, wy)
    TTTT     = x - sol
    obj      = 0.5 * jnp.sum(TTTT**2) + gamma * norm_tv2d(sol, wx, wy)
    rel_obj  = abs(obj - prev_obj) / obj
    prev_obj = obj
    [dx, dy] = gradient_op2d(sol, wx, wy)
    r        = r - 1 / (12 * gamma * mt**2) * dx
    s        = s - 1 / (12 * gamma * mt**2) * dy
    weights  = jnp.maximum(1, jnp.sqrt(jnp.abs(r) ** 2 + jnp.abs(s) ** 2))
    p        = r / weights
    q        = s / weights
    t        = (1 + jnp.sqrt(4 * told**2)) / 2
    r        = p + (told - 1) / t * (p - pold)
    pold     = p
    s        = q + (told - 1) / t * (q - qold)
    qold     = q
    told     = t
    return sol, pold, qold, told, r, s, prev_obj, rel_obj


def prox_tv2d(x, g):
    tol     = 10e-4
    verbose = 1
    maxit   = 200
    weights = jnp.ones(2)
    gamma   = g
    if gamma < 0:
        print("gamma cannot be negative")
    wx       = weights[0]
    wy       = weights[1]
    mt       = jnp.amax(weights)
    [r, s]   = gradient_op2d(x * 0)
    pold     = r
    qold     = s
    told     = 1
    prev_obj = 0
    for iter in jnp.arange(maxit):
        sol, pold, qold, told, r, s, prev_obj, rel_obj = prox_tv2d_step(x, gamma, prev_obj, r, s, told, pold, qold, wx, wy, mt, tol)
        if rel_obj < tol:
            break
    return sol

@partial(jax.vmap, in_axes=(0, 0, None, None))
def initest(f_ill, T, pdar, sig):
    T2 = jnp.pad(jnp.repeat(jnp.repeat(T, sig, axis=0), sig, axis=1),((pdar, pdar), (pdar, pdar)),mode="constant")
    Fk = jnp.exp(1j*2*jnp.pi*(f_ill[0]*X + f_ill[1]*Y))
    Am = jnp.abs(ifft2(fft2(T2 * Fk) * Hoh))
    return Am

@partial(jax.vmap, in_axes=(0, None, None, None))
def Forward(f_ill, T, pdar, sig):
    Fk = jnp.exp(1j*2*jnp.pi*(f_ill[0]*X+f_ill[1]*Y))
    Uk = ifft2(fft2(T*Fk)*Ho)
    An = jnp.sqrt(
        reduce_window(
            jnp.abs(dynamic_slice(Uk, (pdar, pdar), (n, n))) ** 2,
            0,
            add,
            (sig, sig),
            (sig, sig),
            "VALID",
        )
        / sig**2
    )
    return Uk, Fk, An

@jax.jit
def step(f_ill, Obj, Ak_meas):
    [Uk, Fk, Ak_forw] = jit_forward(f_ill, Obj, pdar, sig)
    cost              = jnp.mean((Ak_forw - Ak_meas) ** 2)*sig
    T1                = 0.5 * ((Ak_forw - Ak_meas) / Ak_forw)
    T2                = jnp.where(jnp.isnan(T1) | jnp.isinf(T1), 0, T1)
    T3                = jnp.pad(jnp.repeat(jnp.repeat(T2, sig, axis=1), sig, axis=2),((0, 0), (pdar, pdar), (pdar, pdar)),mode="constant")
    T4                = ifft2(fft2(T3*Uk)*Hoh)*jnp.conj(Fk)
    gradi             = T4.sum(axis=0)
    return gradi, cost,Ak_forw

@jax.jit
def fista_step(params, params_prox, params_prox1, t_prev):
    t_new       = (1 + jnp.sqrt(1 + 4 * t_prev**2)) / 2
    params_new  = params_prox1 + ((t_prev-1)/t_new)*(params_prox1 - params_prox)
    t_prev      = t_new
    params_prox = params_prox1
    return params_new, t_prev, params_prox

@jax.jit
def Objoptfunc(OBJparam, F_P_exp, T, K, Kotf, W):
    sz                   = F_P_exp.shape[0] // 2
    K                    = K.at[sz, sz].set(1)
    Objamp               = OBJparam[0] * (jnp.abs(K) ** OBJparam[1])
    Signalamp            = Objamp * jnp.abs(T)
    Noisespectrum        = F_P_exp * W
    NoisePower           = jnp.sum(jnp.abs(Noisespectrum) ** 2) / jnp.sum(W)
    Noisefreesignalpower = jnp.abs(F_P_exp) ** 2 - NoisePower
    Error                = Noisefreesignalpower - Signalamp**2
    Zloop                = (K < 0.7 * Kotf) * (K > 0.2 * Kotf)
    invK                 = 1.0 / K
    ZZ                   = jnp.sum((Error**2 * invK) * Zloop)
    return ZZ


@jax.jit
def gradient_Objoptfunc(OBJparam, F_P_exp, T, K, Kotf, W):
    return jax.grad(Objoptfunc)(OBJparam, F_P_exp, T, K, Kotf, W)


def Object_Para_Estimation(F_P_exp, T, Kotf, U, V, W):
    K        = jnp.sqrt(U**2 + V**2)
    CC       = (K > 0.2 * Kotf) * (K < 0.7* Kotf)
    NSK      = F_P_exp * CC
    A        = jnp.abs(NSK).sum() / CC.sum()
    alpha    =  -0.5
    OBJparam = jnp.array([A, alpha])
    def scipy_obj(params):
        return Objoptfunc(params, F_P_exp, T, K, Kotf, W).item()
    def scipy_grad(params):
        return np.array(gradient_Objoptfunc(params, F_P_exp, T, K, Kotf, W))
    bounds  = [(0, None), (-2, 2)]
    opt_result = minimize(
        scipy_obj,
        OBJparam,
        jac=scipy_grad,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 500},
    )
    Objpara = opt_result.x
    return Objpara[0], -Objpara[1]

@jax.jit
def Weiner_filter_center(AF1, U, V, A, alpha, Otf, Kotf, K, W, lambdaa):
    Noisespectrum = AF1 * W
    sz            = AF1.shape[0] // 2
    K             = K.at[sz, sz].set(1)
    NoisePower    = jnp.sum(jnp.abs(Noisespectrum) ** 2) / jnp.sum(W)
    Otfpower      = jnp.abs(Otf) ** 2
    OBJamp        = A * jnp.abs(K) ** (-alpha)
    OBJpower      = OBJamp**2
    T1            = jnp.conj(Otf) / NoisePower
    T2            = Otfpower / NoisePower
    T3            = 1.0 / OBJpower
    T4            = T1 / (T2 + T3)
    Filter        = jnp.where(jnp.isnan(T4) | jnp.isinf(T4), 0, T4)
    WF_AF1        = Filter * AF1
    Noise_AF1     = NoisePower
    WF_AF1 = jnp.where(jnp.isnan(WF_AF1) | jnp.isinf(WF_AF1), 0, WF_AF1)
    return WF_AF1, Noise_AF1


@jax.jit
def phase_only(x):
    #return jnp.exp(1j * jnp.angle(x))
    return jnp.where(abs(reconobj) > 1, jnp.exp(1j*jnp.angle(reconobj)), reconobj)


# JIT-compiled versions of the main operators
jit_initest         = jax.jit(initest, static_argnums=(2, 3))
jit_forward         = jax.jit(Forward, static_argnums=(2, 3))
jit_dftregistration = jax.jit(dftregistration, static_argnums=(2, 3))
jit_FTpad           = jax.jit(FTpad)
jit_updateFill      = jax.jit(updateFill)

# Propagation transfer function in Fourier domain
prop_phs = 1j * 2 * jnp.pi * jnp.sqrt(((1 / lambdaa) ** 2 - (Fx**2 + Fy**2)) + 1j * 0)
Ho       = jnp.exp(prop_phs * do)           # forward propagation kernel
Hoh      = jnp.conj(jnp.exp(prop_phs * do)) # back-propagation / adjoint kernel

# NA crop mask (not used directly yet, but defines pupil support)
NA_crop  = (Fx**2 + Fy**2) > (NA / lambdaa) ** 2

# Initial estimate (accumulate intensity over all illuminations)
Init_est = np.zeros((N, N))

# Build intensity-based initial guess by summing simulated measurements
for i in np.arange(0, n_batches):
    EE = jit_initest(
        F_illum[batch_size * i : batch_size * (i + 1)],
        A_acqs[batch_size * i : batch_size * (i + 1)],
        pdar,
        sig,
    )
    Init_est += EE.sum(axis=0)

# Average over number of illuminations
Init_est = Init_est / N_illum

# Normalize for display
Init_est_rescaled = (Init_est - Init_est.min()) / (Init_est.max() - Init_est.min())

# Show initial estimate
plt.ion()
plt.figure()
plt.imshow(Init_est_rescaled, cmap="gray")  # Display rescaled grayscale image
plt.colorbar()
plt.show()

# Initialize reconstruction object (complex field)
reconobj      = np.ones((N, N), dtype=jnp.complex64)
reconobj_prox = reconobj

# Main reconstruction loop (gradient descent + TV + FISTA + angle correction)
total_start = timer()
for iter in np.arange(N_iter):
    grads_agg = jnp.zeros_like(reconobj)   # accumulate gradients over batches

    # Enforce phase-only object for first irer_num iterations
    if iter < irer_num:  # (you were missing the colon here)
        reconobj = jnp.exp(1j * jnp.angle(reconobj))
        # Optionally: phase-only only where |reconobj| > 1
        # reconobj = jnp.where(abs(reconobj) > 1, jnp.exp(1j*jnp.angle(reconobj)), reconobj)

    # Loop over illumination batches in random order
    for i in sample(list(range(n_batches)), n_batches):
        [grads, cost_k, Ak_forw] = step(
            F_illum[batch_size * i : batch_size * (i + 1)],
            reconobj,
            A_acqs[batch_size * i : batch_size * (i + 1)],
        )

        # Angle/shift correction within selected iterations
        if iter > iter_Start - 1 and iter < iter_Stop - 1 and ang_corr == 1:
            # Subpixel registration between forward model and measured amplitudes
            shift_opt = jit_dftregistration(
                Ak_forw,
                A_acqs[batch_size * i : batch_size * (i + 1)],
                usfac,
                no,
            )
            # Update shift estimates and corresponding illumination frequencies
            [Fi_x, Fi_y, si_x, si_y] = jit_updateFill(
                shift_opt,
                Shift_x[batch_size * i : batch_size * (i + 1)],
                Shift_y[batch_size * i : batch_size * (i + 1)],
            )

            # Write updated shifts back into global arrays
            Shift_x = Shift_x.at[batch_size * i : batch_size * (i + 1)].set(si_x)
            Shift_y = Shift_y.at[batch_size * i : batch_size * (i + 1)].set(si_y)

            # Update illumination frequency estimates
            F_illum = F_illum.at[batch_size * i : batch_size * (i + 1), 0].set(Fi_x)
            F_illum = F_illum.at[batch_size * i : batch_size * (i + 1), 1].set(Fi_y)

        # Accumulate gradients and cost
        grads_agg   += grads
        cost[iter]  = cost[iter] + cost_k
        reconobj.block_until_ready()  # ensure computation is finished before next iter

    # Average gradient over all illuminations
    grads_agg = grads_agg / N_illum

    # Gradient descent update
    reconobj = reconobj - step_size * grads_agg
    # Replace NaN/Inf with 1 to avoid numerical blow-up
    reconobj = jnp.where(jnp.isnan(reconobj) | jnp.isinf(reconobj), 1, reconobj)

    # TV proximal step
    reconObj_prox1 = prox_tv2d(reconobj, regparam)

    # Simple FISTA restart if cost increases
    if iter > 1:
        if cost[-1] > cost[-2]:
            t_k      = 1
            reconobj = reconobj_prox
            continue

    # FISTA momentum update
    [reconobj, t_k, reconobj_prox] = fista_step(
        reconobj,
        reconobj_prox,
        reconObj_prox1,
        t_k,
    )

# Final corrected illumination frequencies and shifts
fx_illum_a     = F_illum[:, 0]
fy_illum_a     = F_illum[:, 1]
Shift_x_a      = Shift_x
Shift_y_a      = Shift_y
thetax_illum_a = jnp.arcsin(-fx_illum_a * lambdaa)
thetay_illum_a = jnp.arcsin(-fy_illum_a * lambdaa)

# Plot illumination angle trajectory before vs after correction
plt.figure()
plt.plot(thetax_illum_b, thetay_illum_b, 'r.', label='Without correction')
plt.plot(thetax_illum_a, thetay_illum_a, 'b.', label='With correction')
plt.axis('equal')
plt.axis('tight')
plt.title('Illuminaiton angles after correction')
plt.legend()
plt.show()

# Show reconstructed phase (cropped to remove padding)
plt.figure()
plt.imshow(jnp.angle(reconobj[pdar:-pdar, pdar:-pdar]),
           cmap="gray", vmin=-0.4, vmax=1.4)
plt.colorbar()
plt.show()

# Center index in frequency axis
wo     = int(N / 2)
k      = 2 * jnp.pi / lambdaa

# Recompute frequency step and frequency grid
dfx    = 1 / (N * ps)
u      = dfx * jnp.arange(-N / 2, N / 2)
[U, V] = jnp.meshgrid(u, u)
K      = jnp.sqrt(U**2 + V**2)   # radial spatial frequency

# Extract reconstructed phase and its Fourier transform
P_exp   = jnp.angle(reconobj)
F_P_exp = fftshift(fft2(P_exp))

# Gaussian model G in frequency domain (used as prior / envelope)
G = (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(-(U**2 + V**2) * (1 / (2 * sigma**2)))
G = G / G.max()  # normalize peak to 1

# Normalized log magnitude of experimental phase spectrum
A    = jnp.log(jnp.abs(F_P_exp)) / jnp.max(jnp.log(jnp.abs(F_P_exp)))
Kotf = Kotfval * (1 / lambdaa)    # approximate OTF cutoff frequency
W    = K > W_cut * Kotf           # frequency mask for estimating noise

# Build apodization mask (outer taper) based on distance from support
Apw         = K > Apo_cut * Kotf
DistApoMask = distance_transform_edt(~Apw)      # distance to mask boundary
maxApoMask  = np.max(DistApoMask)
ApoFunc     = (DistApoMask / maxApoMask) ** Index  # smooth roll-off

# Plot 1D cuts of spectral model components along central row
plt.figure()
plt.plot(u, A[wo + 1, :], label='A')               # experimental log spectrum
plt.plot(u, G[wo + 1, :], label='G')               # Gaussian model
plt.plot(u, W[wo + 1, :], label='W')               # noise weighting mask
plt.plot(u, ApoFunc[wo + 1, :], label='ApoFunc')   # apodization function
plt.legend()
plt.title("Power Spectrum of Uniform Illumination")
plt.xlabel("Frequency (u)")
plt.ylabel("Log Magnitude")
plt.grid(True)
plt.axvline(Kotf, color="r", linestyle="--")
plt.show()

# Estimate object spectral parameters (Al, al) via optimization
Al, al = Object_Para_Estimation(F_P_exp, G, Kotf, U, V, W)

print(Al)
print(al)

# Recompute radial frequency grid and estimated signal amplitude spectrum
K      = jnp.sqrt(U**2 + V**2)
Objamp = Al * (K**-al)
Sigamp = Objamp * jnp.abs(G)

# Compare actual vs estimated signal power spectrum
plt.figure()
plt.plot(u, jnp.log(jnp.abs(F_P_exp[wo + 1, :])), "k--", label="Actual signal power")
plt.plot(u, jnp.log(jnp.abs(Sigamp[wo + 1, :])), "m-", label="Estimated signal power")
plt.legend()
plt.title("Power spectrum of uniform illumination")
plt.xlabel("Frequency (u)")
plt.ylabel("Log Magnitude")
plt.grid(True)
plt.show()

# Apply Wiener filter in frequency domain using estimated object + noise model
WF_P_exp, Noise_F_P_exp = Weiner_filter_center(
    F_P_exp, U, V, Al, al, G, Kotf, K, W, lambdaa
)

# Inverse FFT of Wiener-filtered phase spectrum (denoised phase)
WP_exp = jnp.real(ifft2(ifftshift(WF_P_exp)))
WP_exp = jnp.where(jnp.isnan(WP_exp) | jnp.isinf(WP_exp), 1, WP_exp)

# Apply apodization to Wiener-filtered spectrum and invert
Imfg = jnp.real(ifft2(ifftshift(WF_P_exp * ApoFunc)))
Imfg = jnp.where(jnp.isnan(Imfg) | jnp.isinf(Imfg), 1, Imfg)

# Print total runtime in minutes
total_end = timer()
print(f"Total time = {(total_end - total_start)/60}")

# Show final filtered phase image
plt.figure()
plt.imshow(Imfg, cmap="gray", vmin=-0.4, vmax=1.4)
plt.colorbar()
plt.title("Filtered Phase Image")
plt.show()

# Log spectra of raw phase, Wiener phase, and apodized-Wiener phase
A = jnp.log(jnp.abs(fftshift(fft2(P_exp)))) / jnp.max(
    jnp.log(jnp.abs(fftshift(fft2(P_exp))))
)
B = jnp.log(jnp.abs(fftshift(fft2(WP_exp)))) / jnp.max(
    jnp.log(jnp.abs(fftshift(fft2(WP_exp))))
)
C = jnp.log(jnp.abs(fftshift(fft2(Imfg)))) / jnp.max(
    jnp.log(jnp.abs(fftshift(fft2(Imfg))))
)

# Plot power spectra comparison and apodization profile
plt.figure()
plt.plot(u, A[wo + 1, :], linewidth=0.5, label="Raw phase")
plt.plot(u, B[wo + 1, :], linewidth=0.5, label="Wiener phase")
plt.plot(u, C[wo + 1, :], linewidth=0.5, label="Apodized Wiener")
plt.plot(u, ApoFunc[wo + 1, :], label="ApoFunc")
plt.legend()
plt.title("Power spectrum of uniform illumination")
plt.xlabel("Frequency (u)")
plt.ylabel("Log Magnitude")
plt.grid(True)
plt.axvline(Kotf, color="r", linestyle="--")
plt.show()

# Pack outputs + metadata to save for MATLAB / further analysis
data_dict = {
    "P_exp": P_exp,
    "WP_exp": WP_exp,
    "imfg": Imfg,
    "A_acqs": A_acqs,
    "N": N,
    "do": data["do"].item(),
    "dpix": data["dpix"].item(),
    "pdar": data["pdar"].item(),
    "m": m,
    "sig": sig,
    "n": n,
    "cx": data["cx"].item(),
    "cy": data["cy"].item(),

    # Original corrected calibration
    "fx_illum_C": jnp.squeeze(data["fx_illum_C"]),
    "fy_illum_C": jnp.squeeze(data["fy_illum_C"]),
    "Shift_x_C": jnp.squeeze(data["Shift_x_C"]),
    "Shift_y_C": jnp.squeeze(data["Shift_y_C"]),
    "thetax_illum_C": jnp.squeeze(data["thetax_illum_C"]),
    "thetay_illum_C": jnp.squeeze(data["thetay_illum_C"]),

    # Before correction (b)
    "fx_illum_b": fx_illum_b,
    "fy_illum_b": fy_illum_b,
    "thetax_illum_b": thetax_illum_b,
    "thetay_illum_b": thetay_illum_b,
    "Shift_x_b": Shift_x_b,
    "Shift_y_b": Shift_y_b,

    # After correction (a)
    "fx_illum_a": fx_illum_a,
    "fy_illum_a": fy_illum_a,
    "thetax_illum_a": thetax_illum_a,
    "thetay_illum_a": thetay_illum_a,
    "Shift_x_a": Shift_x_a,
    "Shift_y_a": Shift_y_a,
}

# Save everything to a .mat file for later analysis / comparison in MATLAB
savemat(filename, data_dict)
