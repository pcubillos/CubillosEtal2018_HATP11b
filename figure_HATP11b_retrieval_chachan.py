#! /usr/bin/env python

import sys
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf

sys.path.append("./BART/modules/transit/transit/python")
import transit_module as tm
sys.path.append("./BART/code")
import makeatm as ma
import wine as w
import bestFit as bf
sys.path.append("./inputs/ancil")
import balance as b
matplotlib.use('tkagg')


# Fraine data:
fraine_wl = np.array([
    1.153, 1.172, 1.190, 1.209, 1.228, 1.247, 1.266, 1.284, 1.303, 1.322,
    1.341, 1.360, 1.379, 1.397, 1.416, 1.435, 1.454, 1.473, 1.492, 1.510,
    1.529, 1.548, 1.567, 1.586, 1.604, 1.623, 1.642, 1.661, 1.680,
    3.600, 4.500])
fraine_depth = np.array([
    0.003502, 0.003407, 0.003421, 0.003445, 0.003350, 0.003377, 0.003380,
    0.003457, 0.003436, 0.003448, 0.003476, 0.003536, 0.003499, 0.003498,
    0.003524, 0.003591, 0.003524, 0.003520, 0.003447, 0.003344, 0.003513,
    0.003471, 0.003438, 0.003414, 0.003383, 0.003415, 0.003480, 0.003498,
    0.003376, 0.003354, 0.003373])
fraine_uncert = np.array([
    0.000040, 0.000047, 0.000046, 0.000038, 0.000041, 0.000038, 0.000045,
    0.000040, 0.000035, 0.000039, 0.000043, 0.000044, 0.000046, 0.000040,
    0.000041, 0.000044, 0.000043, 0.000039, 0.000039, 0.000045, 0.000041,
    0.000050, 0.000049, 0.000053, 0.000045, 0.000038, 0.000048, 0.000060,
    0.000074, 0.000025, 0.000029])


chachan_depth = np.array([
    0.003350,     0.003388,     0.003397,     0.003378,     0.003485,
    0.003271,     0.003349,     0.003325,     0.003121,     0.003321,
    0.003351,     0.003286,     0.003133,     0.003234,     0.003378,
    0.003339,     0.003343,     0.003358,     0.003372,     0.003376,
    0.003370,     0.003345,     0.003377,     0.003349,     0.003377,
    0.003400,     0.003480,     0.003476,     0.003393,     0.003295,
    0.003279,     0.003413,     0.003367,     0.003498,     0.003442,
    0.003492,     0.003530,     0.003520,     0.003307,     0.003455,
    0.003418,     0.003395,     0.003468,     0.003540,     0.003391,
    0.003338,     0.003377,
    ])

chachan_uncert = np.array([
    0.000135,     0.000052,     0.000036,     0.000034,     0.000102,
    0.000078,     0.000081,     0.000084,     0.000122,     0.000097,
    0.000134,     0.000148,     0.000178,     0.000239,     0.000022,
    0.000018,     0.000017,     0.000016,     0.000015,     0.000015,
    0.000015,     0.000013,     0.000015,     0.000014,     0.000014,
    0.000014,     0.000052,     0.000048,     0.000033,     0.000038,
    0.000036,     0.000027,     0.000027,     0.000035,     0.000036,
    0.000035,     0.000037,     0.000036,     0.000033,     0.000032,
    0.000035,     0.000042,     0.000035,     0.000044,     0.000119,
    0.000029,     0.000032,
    ])

chachan_wl = np.array([
    0.373, 0.428, 0.484, 0.538, 0.552, 0.601, 0.650, 0.699, 0.748, 0.796,
    0.845, 0.895, 0.943, 0.992, 0.861, 0.885, 0.909, 0.931, 0.955, 0.978,
    1.002, 1.025, 1.048, 1.071, 1.095, 1.119, 1.135, 1.165, 1.195, 1.225,
    1.255, 1.285, 1.315, 1.345, 1.375, 1.405, 1.435, 1.465, 1.495, 1.525,
    1.555, 1.585, 1.615, 1.645, 1.675, 3.600, 4.500,
    ])


filters = [
    'inputs/filters/chachan2019_stis_g430l_0.373um.dat',
    'inputs/filters/chachan2019_stis_g430l_0.428um.dat',
    'inputs/filters/chachan2019_stis_g430l_0.484um.dat',
    'inputs/filters/chachan2019_stis_g430l_0.538um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.552um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.601um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.650um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.699um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.748um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.796um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.845um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.895um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.943um.dat',
    'inputs/filters/chachan2019_stis_g750l_0.992um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.861um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.885um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.909um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.931um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.955um.dat',
    'inputs/filters/chachan2019_wfc3_g102_0.978um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.002um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.025um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.048um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.071um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.095um.dat',
    'inputs/filters/chachan2019_wfc3_g102_1.119um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.135um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.165um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.195um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.225um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.255um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.285um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.315um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.345um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.375um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.405um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.435um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.465um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.495um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.525um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.555um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.585um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.615um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.645um.dat',
    'inputs/filters/chachan2019_wfc3_g141_1.675um.dat',
    'BART/inputs/filters/spitzer_irac1_sa.dat',
    'BART/inputs/filters/spitzer_irac2_sa.dat',
    ]

# Retrieval outputs from local run:
root = "./run07_HAT-P-11b_BART/retrieval_chachan_all/"
besttcfg  = root + "bestFit_tconfig.cfg"  # Transit cfg file

# Transit configuration file for best-fit:
args = ["transit", "-c", besttcfg]
tm.transit_init(len(args), args)
wn = tm.get_waveno_arr(tm.get_no_samples())
wl = 1e4/wn

# Read filters:
filter_wn, filter_tr, filter_interp, filter_idx = [], [], [], []
for filt in filters:
    fwn, ftr = w.readfilter(filt)
    filter_wn.append(fwn)
    filter_tr.append(ftr)
    ifilter, _, idx = w.resample(wn, fwn, ftr, None, None)
    filter_interp.append(ifilter)
    filter_idx.append(idx)

irac1_tr = filter_tr[-2] - np.amin(filter_tr[-2])
irac2_tr = filter_tr[-1] - np.amin(filter_tr[-1])
irac1_tr /= np.amax(irac1_tr)
irac2_tr /= np.amax(irac2_tr)
irac1_wl, irac2_wl = 1e4/filter_wn[-2], 1e4/filter_wn[-1]

# atmospheric model:
mol, press, temp, abund_init = ma.readatm(
    root + "atmosphere_HAT-P-11b_uniform.atm")

bestp_all, bestu_all = bf.read_MCMC_out(
    "./run07_HAT-P-11b_BART/retrieval_chachan_all/MCMC_chachan_all.log")
bestT_all, bestrad_all, bestcl_all, bestray_all = bestp_all[0:4]
bestabund_all = bestp_all[5:]

bestp_hst, bestu_hst = bf.read_MCMC_out(
    "./run07_HAT-P-11b_BART/retrieval_chachan_hst/MCMC_chachan_hst.log")
bestT_hst, bestrad_hst, bestcl_hst, bestray_hst = bestp_hst[0:4]
bestabund_hst = bestp_hst[5:]

# Initialize stuff:
imol = [0,1]
na = np.copy(abund_init)
rat, irat = b.ratio(na, imol)
profiles = np.vstack((temp, abund_init.T))

# Best (HST + Spitzer):
temp[:] = bestT_all
tm.set_radius(bestrad_all)
tm.set_cloudtop(bestcl_all)
tm.set_scattering(bestray_all)
na = np.copy(abund_init)
rat, irat = b.ratio(na, imol)
na[:, 2:] *= 10**bestabund_all
b.balance(na, imol, rat, irat)
profiles = np.vstack((temp, na.T))
best_spectrum_all = tm.run_transit(profiles.flatten(), tm.get_no_samples())

# Best (HST):
temp[:] = bestT_hst
tm.set_radius(bestrad_hst)
tm.set_cloudtop(bestcl_hst)
tm.set_scattering(bestray_hst)
na = np.copy(abund_init)
rat, irat = b.ratio(na, imol)
na[:, 2:] *= 10**bestabund_hst
b.balance(na, imol, rat, irat)
profiles = np.vstack((temp, na.T))
best_spectrum_hst = tm.run_transit(profiles.flatten(), tm.get_no_samples())

sigma = 6
mod_all = gaussf(best_spectrum_all, sigma)  # Best
mod_hst = gaussf(best_spectrum_hst, sigma)  # Solar

band_all = np.array([
    w.bandintegrate(best_spectrum_all[wnidx], wn, ifl, wnidx)
    for ifl,wnidx in zip(filter_interp, filter_idx)])
band_hst = np.array([
    w.bandintegrate(best_spectrum_hst[wnidx], wn, ifl, wnidx)
    for ifl,wnidx in zip(filter_interp, filter_idx)])


fs = 12
lw = 1.0
f = 1e6  # ppm
yran = f*np.array([0.00294, 0.00406])

plt.figure(4, (8.5,3.5))
plt.clf()
plt.subplots_adjust(0.10, 0.15, 0.99, 0.98)
ax = plt.subplot(111)

plt.semilogx(wl, f*mod_hst, c="cornflowerblue", lw=lw, label="HST")
plt.semilogx(wl, f*mod_all, c="salmon", lw=lw, label="HST + Spitzer")
plt.plot(chachan_wl[-2:], f*band_hst[-2:], "o", ms=4, c='blue', zorder=3)
plt.plot(chachan_wl[-2:], f*band_all[-2:], "o", ms=4, c='red', zorder=3)

plt.errorbar(chachan_wl, f*chachan_depth, f*chachan_uncert, fmt="ok", ms=4,
    elinewidth=lw, zorder=4, capthick=lw, label='Chachan et al. (2019)')
plt.errorbar(fraine_wl, f*fraine_depth, f*fraine_uncert, fmt="o", ms=4,
    color='0.7', elinewidth=lw, zorder=0, capthick=lw,
    label='Fraine et al. (2014)')

plt.plot(irac1_wl, 100*irac1_tr + yran[0], color="0.5", lw=lw)
plt.plot(irac2_wl, 100*irac2_tr + yran[0], color="0.5", lw=lw)
leg = plt.legend(loc="upper left", fontsize=fs-2)
plt.xlabel(r"Wavelength (um)", fontsize=fs)
plt.ylabel(r"$(R_{\rm p}/R_{\rm s})^2$ (ppm)", fontsize=fs)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([0.4, 0.7, 1.0, 1.4, 2, 3, 5])
plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.xlim(0.34, 5.5)
plt.ylim(yran)
plt.savefig("plots/HATP11b_spectra_chachan.pdf")


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# indices in posterior array:
itemp, irad, icloud, iray, iH2O, iCH4, iCO, iCO2 = np.arange(8)

km = 1e3
rearth = 6.3781e6
rjup = 7.1492e7
rsun = 6.96e8

burn = 8000

sample_all = np.load("run07_HAT-P-11b_BART/retrieval_chachan_all/output.npy")
temp_all  = sample_all[:,itemp,burn:].flatten()
cloud_all = sample_all[:,icloud,burn:].flatten()
ray_all   = sample_all[:,iray,burn:].flatten()
rad_all   = sample_all[:,irad,burn:].flatten() * km / rearth
H2O_all   = sample_all[:,iH2O,burn:].flatten() - 10.0
CH4_all   = sample_all[:,iCH4,burn:].flatten() - 10.0
CO_all    = sample_all[:,iCO ,burn:].flatten() - 10.0
CO2_all   = sample_all[:,iCO2,burn:].flatten() - 10.0

sample_hst = np.load("run07_HAT-P-11b_BART/retrieval_chachan_hst/output.npy")
temp_hst  = sample_hst[:,itemp,burn:].flatten()
cloud_hst = sample_hst[:,icloud,burn:].flatten()
ray_hst   = sample_hst[:,iray,burn:].flatten()
rad_hst   = sample_hst[:,irad,burn:].flatten() * km / rearth
H2O_hst   = sample_hst[:,iH2O,burn:].flatten() - 10.0
CH4_hst   = sample_hst[:,iCH4,burn:].flatten() - 10.0
CO_hst    = sample_hst[:,iCO ,burn:].flatten() - 10.0
CO2_hst   = sample_hst[:,iCO2,burn:].flatten() - 10.0


posterior_all = np.vstack([
    temp_all, rad_all, cloud_all, ray_all,
    H2O_all, CH4_all, CO_all, CO2_all]).T

posterior_hst = np.vstack([
    temp_hst, rad_hst, cloud_hst, ray_hst,
    H2O_hst, CH4_hst, CO_hst, CO2_hst]).T

median_pars_all = np.median(posterior_all, axis=0)
plow_all = np.percentile(posterior_all, 15.865, axis=0)
phigh_all = np.percentile(posterior_all, 84.135, axis=0)

median_pars_hst = np.median(posterior_hst, axis=0)
plow_hst = np.percentile(posterior_hst, 15.865, axis=0)
phigh_hst = np.percentile(posterior_hst, 84.135, axis=0)

dec = [0, 2, 1, 1, 1, 1, 1, 1]
for hmed,hlo,hhi, amed,alo,ahi, d in zip(
        median_pars_hst, plow_hst, phigh_hst,
        median_pars_all, plow_all, phigh_all, dec):
    hlow = hmed-hlo
    hhigh = hhi-hmed
    alow = amed-alo
    ahigh = ahi-amed
    print(f'${hmed:.{d}f}_{{-{hlow:.{d}f}}}^{{+{hhigh:.{d}f}}}$  &  '
          f'${amed:.{d}f}_{{-{alow:.{d}f}}}^{{+{ahigh:.{d}f}}}$')


pnames = [
    r"$T$ (K)",
    r"$R_{\rm p}$ ($R_{\oplus}$)",
    r"$\log_{10}(\frac{P_{\rm cloud}}{{\rm bar}})$",
    r"$\log_{10}(\kappa_{\rm ray})$",
    r"$\log_{10}({\rm H2O})$",
    r"$\log_{10}({\rm CH4})$",
    r"$\log_{10}({\rm CO})$",
    r"$\log_{10}({\rm CO2})$"]


nsamples, npars = np.shape(posterior_all)

ranges = [
    [500, 2000], [3.6, 4.6], [-5.0, 0.5], [-5.0,5.0],
    [-6, 0], [-10,0], [-10,0], [-10,0]]

palette = copy.copy(plt.cm.RdPu)
palette.set_under(color='w')
palette.set_bad(color='w')

hist = []
hist2 = []
xran, yran, lmax = [], [], []
for irow in range(1, npars):
    for icol in range(irow):
        ran = None
        if ranges[icol] is not None:
            ran = [ranges[icol], ranges[irow]]
        h, x, y = np.histogram2d(posterior_all[:,icol], posterior_all[:,irow],
            bins=16, range=ran)
        h2,x2,y2= np.histogram2d(posterior_hst[:,icol], posterior_hst[:,irow],
            bins=16, range=ran)
        hist.append(h.T)
        hist2.append(h2.T)
        xran.append(x)
        yran.append(y)
        lmax.append(np.amax(h)+1)

# Mean molecular masses:
Zsun = 0.0134
Xsun = 0.7381
QX_over_QY = 0.85/0.15  # By volume

species = ['H2', 'He', 'H2O', 'CO', 'CO2', 'CH4']
nspec = len(species)
mass = np.array([2.01588, 4.002602, 18.01528, 28.0101, 44.0095, 16.0425])
H_mass = 1.00794
He_mass = mass[1]

nq = 1000
q = np.zeros((3,nq))
q[2] = np.logspace(-6, 0, nq)
q[1] = (1-q[2]) / (1.0 + QX_over_QY)
q[0] = QX_over_QY * q[1]
mu = np.sum(q*np.expand_dims([2.01588, 4.002602, 18.01528],axis=1),axis=0)
y = q[1] * He_mass
x = H_mass * 2.0 * (q[0] + q[2])
z = mu - x - y
mass_fraction = np.log10(z/x / (Zsun/Xsun))
mu_ticks = 2.32, 5.0
mf_ticks = [mass_fraction[mu>=mu_tick][0] for mu_tick in mu_ticks]


u_hst, uind_hst, uinv_hst = np.unique(
    H2O_hst, return_index=True, return_inverse=True)
nunique_hst = len(u_hst)

Q_hst = np.zeros((nspec, nunique_hst))
Q_hst[2] = 10**H2O_hst[uind_hst]
Q_hst[3] = 10**CO_hst[uind_hst]
Q_hst[4] = 10**CO2_hst[uind_hst]
Q_hst[5] = 10**CH4_hst[uind_hst]
Q_hst[1] = (1-np.sum(Q_hst[2:], axis=0)) / (1.0 + QX_over_QY)
Q_hst[0] = QX_over_QY * Q_hst[1]

# Mass fractions:
mu_hst = np.sum(Q_hst * np.expand_dims(mass,axis=1), axis=0)
Y_hst = Q_hst[1] * He_mass
X_hst = H_mass * (2*Q_hst[0] + 2*Q_hst[2] + 4*Q_hst[5])
Z_hst = mu_hst - X_hst - Y_hst
mass_fraction_hst = np.log10(Z_hst/X_hst / (Zsun/Xsun))


u_all, uind_all, uinv_all = np.unique(
    H2O_all, return_index=True, return_inverse=True)
nunique_all = len(u_all)

Q_all = np.zeros((nspec,nunique_all))
Q_all[2] = 10**H2O_all[uind_all]
Q_all[3] = 10**CO_all[uind_all]
Q_all[4] = 10**CO2_all[uind_all]
Q_all[5] = 10**CH4_all[uind_all]
Q_all[1] = (1-np.sum(Q_all[2:], axis=0)) / (1.0 + QX_over_QY)
Q_all[0] = QX_over_QY * Q_all[1]

# Mass fractions:
mu_all = np.sum(Q_all*np.expand_dims(mass,axis=1), axis=0)
Y_all = Q_all[1] * He_mass
X_all = H_mass * (2*Q_all[0] + 2*Q_all[2] + 4*Q_all[5])
Z_all = mu_all - X_all - Y_all
mass_fraction_all = np.log10(Z_all/X_all / (Zsun/Xsun))

# Plot:
nlevels = 20
fs = 11


plt.figure(201, figsize=(8.5,8.5))
plt.clf()
plt.subplots_adjust(0.08, 0.09, 0.99, 0.99, hspace=0.1, wspace=0.1)
axes = np.tile(None, (npars, npars))
k = 0 # Histogram index
for irow in range(1, npars):
    for icol in range(irow):
        h = (npars)*(irow) + icol + 1  # Subplot index
        ax = plt.subplot(npars, npars, h)
        ax.tick_params(labelsize=fs-1, direction='in')
        if icol == 0:
            ax.set_ylabel(pnames[irow], size=fs)
        else:
            ax.get_yaxis().set_visible(False)
        if irow == npars-1:
            ax.set_xlabel(pnames[icol], size=fs)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        else:
            ax.get_xaxis().set_visible(False)
        cont = ax.contourf(hist[k], cmap=palette, vmin=1, origin='lower',
            levels=[0]+list(np.linspace(1,lmax[k], nlevels)),
            extent=(xran[k][0], xran[k][-1], yran[k][0], yran[k][-1]))
        xx = 0.5*(xran[k][:-1] + xran[k][1:])
        yy = 0.5*(yran[k][:-1] + yran[k][1:])
        X, Y = np.meshgrid(xx, yy)
        CS = ax.contour(X, Y, hist2[k],
            levels=3,
            linewidths=1.0, colors=plt.cm.Blues(np.linspace(0.3, 0.9, 4)))
        for c in cont.collections:
            c.set_edgecolor("face")
        if ranges[icol] is not None:
            ax.set_xlim(ranges[icol])
        if ranges[icol] is not None:
            ax.set_ylim(ranges[irow])
        k += 1

# Histograms
nb = 14
for i in range(npars):
    h = i*(npars+1) + 1
    ax = plt.subplot(npars, npars, h)
    ax.tick_params(labelsize=fs-1, direction='in')
    ax.set_yticks([])
    h, hx, patch = plt.hist(
        posterior_hst[:,i], bins=nb, range=ranges[i], histtype='stepfilled',
        color='cornflowerblue', label='Chachan: HST')
    h, hx, patch = plt.hist(
        posterior_all[:,i], bins=nb, range=ranges[i], histtype='stepfilled',
        color='salmon', alpha=0.7, label='Chachan: HST+Spitzer')
    ax.set_xlim(ranges[i])
    ax.axvline(median_pars_hst[i], color='b', lw=1.5, dashes=())
    ax.axvline(median_pars_all[i], color='r', lw=1.5, dashes=())
    if i != npars-1:
        ax.set_xticklabels("")
    else:
        plt.xlabel(pnames[i], fontsize=fs)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    if i == 0:
        plt.legend(loc=(1.1, 0.45), fontsize=fs)

# Mean molecular mass histogram:
ax = plt.subplot(npars, npars, 14)
ax.hist(
    mass_fraction_hst[uinv_hst], bins=nb,
    histtype='stepfilled', color='cornflowerblue')
ax.hist(
    mass_fraction_all[uinv_all], bins=nb,
    histtype='stepfilled', color='salmon', alpha=0.7)
ax.set_xlabel('$[M_Z/M_X]$', size=fs)
ax.set_yticks([])
ax.axvline(np.median(mass_fraction_hst[uinv_hst]), color='b', lw=1.5)
ax.axvline(np.median(mass_fraction_all[uinv_all]), color='r', lw=1.5)
ax.set_xlim(-2.6, 2.6)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(mf_ticks)
ax2.set_xticklabels(mu_ticks)
ax2.set_xlabel('Mean molecular mass', fontsize=fs)

# The colorbar:
bounds = np.linspace(0, 1.0, nlevels)
norm = matplotlib.colors.BoundaryNorm(bounds, palette.N)
ax2 = plt.axes([0.94, 0.63, 0.025, 0.36])
cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
    spacing='proportional', boundaries=bounds, format='%.1f')
cb.set_label("Normalized Point Density", fontsize=fs)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')
cb.ax.tick_params(labelsize=fs-1)
cb.set_ticks(np.linspace(0, 1, 5))
for c in ax2.collections:
    c.set_edgecolor("face")
ax3 = plt.axes([0.965, 0.63, 0.025, 0.36])
xx = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xx, xx)
ax3.contourf(X, Y, Y, levels=3,
    colors=plt.cm.Blues(np.linspace(0.3, 0.9, 4)))
ax3.set_ylim(0,1)
ax3.set_yticks([])
ax3.set_xticks([])
plt.savefig('plots/HATP11b_posteriors_chachan.pdf')

