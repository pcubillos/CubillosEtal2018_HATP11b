#! /usr/bin/env python
import sys, os, time
from scipy.ndimage.filters import gaussian_filter1d as gaussf

sys.path.append("./BART/modules/transit/transit/python")
import transit_module as tm
sys.path.append("./BART/code")
import makeatm as ma
import wine as w
import bestFit as bf
sys.path.append("./inputs/ancil")
import balance as b

sys.path.append("/home/patricio/ast/esp01/code/lib/c/pt/lib")
import PT as pt

# HST data:
wlength = np.array([
  1.153, 1.172, 1.190, 1.209, 1.228, 1.247, 1.266, 1.284, 1.303, 1.322,
  1.341, 1.360, 1.379, 1.397, 1.416, 1.435, 1.454, 1.473, 1.492, 1.510,
  1.529, 1.548, 1.567, 1.586, 1.604, 1.623, 1.642, 1.661, 1.680])
depth = 1e-6 * np.array([
  3502, 3407, 3421, 3445, 3350, 3377, 3380, 3457, 3436, 3448, 3476,
  3536, 3499, 3498, 3524, 3591, 3524, 3520, 3447, 3344, 3513, 3471,
  3438, 3414, 3383, 3415, 3480, 3498, 3376.])
uncert = 1e-6 * np.array([40, 47, 46, 38, 41, 38, 45, 40, 35, 39, 43, 44, 46,
              40, 41, 44, 43, 39, 39, 45, 41, 50, 49, 53, 45, 38, 48, 60, 74.])

# Spitzer data:
swlength = np.array([3.6, 4.5])
sdepth  = np.array([0.003354, 0.003373])
suncert = np.array([0.000025, 0.000029])

# Retrieval outputs from Cubillos et al. (2018):
root = "./inputs/bestfit/"
# Retrieval outputs from local run:
#root = "./run07_HAT-P-11b_BART/run07_HAT-P-11b_BART/BART_retrieval/"

besttcfg  = root + "bestFit_tconfig.cfg"  # Transit cfg file
bestatm   = root + "bestFit.atm"          # Atmospheric file
posterior = root + "output.npz"           # Posterior distribution

# Transit configuration file for best-fit:
args = ["transit", "-c", besttcfg]
tm.transit_init(len(args), args)

# Read Spitzer Filters:
irac1_wn, irac1_tr = w.readfilter("BART/inputs/filters/spitzer_irac1_sa.dat")
irac2_wn, irac2_tr = w.readfilter("BART/inputs/filters/spitzer_irac2_sa.dat")
irac1_tr -= np.amin(irac1_tr)
irac2_tr -= np.amin(irac2_tr)
irac1_tr /= np.amax(irac1_tr)
irac2_tr /= np.amax(irac2_tr)
irac1_wl = 1e4/irac1_wn
irac2_wl = 1e4/irac2_wn

# Read atmospheric files:
mol, press, temp, abund = ma.readatm(bestatm) # Best-fit
# Initial guess:
mol2, press2, temp2, abund2 = ma.readatm("./run07_HAT-P-11b_BART/BARTinputs/atmosphe_HAT-P-11b.atm")

sample = np.load(posterior)
# Best-fitting PT parameters:
bestPT    = sample["bestp"][0:5]
# Best-fitting Radius at 0.1 bar:
bestrad   = sample["bestp"][5]
# Best-fitting abundances: H2O CH4 CO CO2
bestabund = sample["bestp"][7:]

# Temperature posterior:
burn = 3000
p0 = sample["Z"][:,0,burn:].flatten()
p1 = sample["Z"][:,1,burn:].flatten()
p2 = sample["Z"][:,2,burn:].flatten()

# Cubillos et al (2016) best PT parameters:
tb = np.zeros(len(press))

Rs, Ts, sma, gstar = bf.get_starData("inputs/TEP/HAT-P-11b.tep")
grav, Rp = ma.get_g("inputs/TEP/HAT-P-11b.tep")
tb = pt.PT_line(press, bestPT, Rs, Ts, 100.0, sma, grav*100)
tprofiles = np.zeros((np.size(p0), len(press)))
Tparams = np.array([-2.8,  -0.55, 1.0, 0.0, 0.965])

for i in np.arange(np.size(p0)):
  Tparams[0] = p0[i]
  Tparams[1] = p1[i]
  Tparams[4] = p2[i]
  tprofiles[i] = pt.PT_line(press, Tparams, Rs, Ts, 100, sma, grav*100)
  if (i+1)%int(np.size(p0)/10) == 0:
    print(i)

low1 = np.percentile(tprofiles, 16,   axis=0)
hi1  = np.percentile(tprofiles, 84,   axis=0)
low2 = np.percentile(tprofiles,  2.5, axis=0)
hi2  = np.percentile(tprofiles, 97.5, axis=0)
median = np.median(tprofiles, axis=0)

# Radius posterior at 0.1 bar:
rad     = sample["Z"][:,3,burn:].flatten()
# WFC3 offset (ppm):
poffset = sample["Z"][:,4,burn:].flatten() * 1e6
# abundances at 0.1 bar:
ipress = 44
pH2O    = sample["Z"][:,5,burn:].flatten() + np.log10(abund2[ipress][9])
pCH4    = sample["Z"][:,6,burn:].flatten() + np.log10(abund2[ipress][8])
pCO     = sample["Z"][:,7,burn:].flatten() + np.log10(abund2[ipress][6])
pCO2    = sample["Z"][:,8,burn:].flatten() + np.log10(abund2[ipress][7])

# Initialize stuff:
imol = [1,5]
na = np.copy(abund)
rat, irat = b.ratio(na, imol)
profiles = np.vstack((temp, abund.T))
wn = tm.get_waveno_arr(tm.get_no_samples())
wl = 1e4/wn

# Best:
tm.set_radius(bestrad)
na = np.copy(abund2)
rat, irat = b.ratio(na, imol)
na[:, 9] *= 10**bestabund[0]
na[:, 8] *= 10**bestabund[1]
na[:, 6] *= 10**bestabund[2]
na[:, 7] *= 10**bestabund[3]
b.balance(na, imol, rat, irat)
profiles = np.vstack((temp, na.T))
tmp2 = tm.run_transit(profiles.flatten(), tm.get_no_samples())

tm.set_radius(29600.0)  # Solar
na = np.copy(abund2)
rat, irat = b.ratio(na, imol)
b.balance(na, imol, rat, irat)
profiles = np.vstack((temp2, na.T))
tmp3 = tm.run_transit(profiles.flatten(), tm.get_no_samples())

tcold = temp2*0.4
tm.set_radius(29900.0)  # Cold
na = np.copy(abund2)
rat, irat = b.ratio(na, imol)
na[:, 9] *= 10**0.0   # H2O
na[:, 8] *= 10**0.0   # CH4
na[:, 6] *= 10**0.0   # CO
na[:, 7] *= 10**0.0   # CO2
b.balance(na, imol, rat, irat)
profiles = np.vstack((tcold, na.T))
tmp4 = tm.run_transit(profiles.flatten(), tm.get_no_samples())

fs = 12
lw = 1.5
yran = 0.0031, 0.00385

sigma = 6
mod2 = gaussf(tmp2, sigma)  # Best
mod3 = gaussf(tmp3, sigma)  # Solar
mod4 = gaussf(tmp4, sigma)

offset = 1.2142876e-04   # SET
plt.figure(4, (8.5,3.5))
plt.clf()
plt.subplots_adjust(0.13, 0.15, 0.95, 0.95)
ax = plt.subplot(111)
plt.semilogx(wl, mod3, "darkorange", lw=lw, label=r"${\rm Solar}$")
plt.semilogx(wl, mod4, "limegreen",  lw=lw, label=r"${\rm Low\ temp.}$")
plt.semilogx(wl, mod2, "b",          lw=lw, label=r"${\rm Best\ fit}$")
plt.errorbar(wlength,  depth-offset, uncert, fmt="or", ms=5, elinewidth=lw,
             zorder=3, capthick=lw)
plt.errorbar(swlength, sdepth,      suncert, fmt="or", ms=5, elinewidth=lw,
             zorder=3, capthick=lw, label=r"${\rm Data}$")
plt.plot(irac1_wl, 0.0001*irac1_tr + yran[0], color="0.5", lw=lw)
plt.plot(irac2_wl, 0.0001*irac2_tr + yran[0], color="0.5", lw=lw)
leg = plt.legend(loc="upper left", fontsize=fs-1)
leg.get_frame().set_alpha(0.5)
plt.xlabel(r"${\rm Wavelength\ \ (um)}$", fontsize=fs)
plt.ylabel(r"$(R_p/R_s)^2$",              fontsize=fs)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([1, 2, 3, 4, 5])
plt.xlim(1.0, 5.5)
plt.ylim(yran)
plt.savefig("plots/HAT-P-11b_spectra.ps")

fs=11
lw=1.5
mname = ["", r"${\rm He}$", "", "", "", r"${\rm H}_2$",
       r"${\rm CO}$", r"${\rm CO}_2$", r"${\rm CH}_4$", r"${\rm H}_2{\rm O}$",
       r"${\rm NH}_3$", r"${\rm C}_2{\rm H}_2$", "", r"${\rm HCN}$"]
matplotlib.rcParams.update({'xtick.minor.size':0.0})
matplotlib.rcParams.update({'ytick.minor.size':0.0})
plt.figure(5, (8.5,3.5))
plt.clf()
ax = plt.axes([0.08, 0.18, 0.3, 0.75])
plt.loglog(abund2[:, 5], press, "-",  lw=lw, label=mname[ 5], color="blue")
plt.loglog(abund2[:, 1], press, "-",  lw=lw, label=mname[ 1],color="darkorange")
plt.loglog(abund2[:, 6], press, "-",  lw=lw, label=mname[ 6], color="limegreen")
plt.loglog(abund2[:, 7], press, "-",  lw=lw, label=mname[ 7], color="r")
plt.loglog(abund2[:, 8], press, "-",  lw=lw, label=mname[ 8], color="m")
plt.loglog(abund2[:, 9], press, "-",  lw=lw, label=mname[ 9], color="k")
plt.loglog(abund2[:,10], press, "--", lw=lw, label=mname[10], color="brown")
plt.loglog(abund2[:,11], press, "--", lw=lw, label=mname[11], color="c")
plt.loglog(abund2[:,13], press, "--", lw=lw, label=mname[13], color="0.4")
plt.legend(loc="lower left", fontsize=fs-1)
plt.ylim(press[0], press[-1])
plt.xlim(1e-26, 1e1)
ax.set_xticks([1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e-0])
ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e-0, 1e2])
plt.xlabel(r"${\rm Mole\ mixing\ ratio}$",     fontsize=fs)
plt.ylabel(r"${\rm Pressure\ \ (bar)}$", fontsize=fs)
matplotlib.rcParams.update({'xtick.minor.size':3.5})

ax = plt.axes([0.4, 0.18, 0.17, 0.75])
ax.fill_betweenx(press, low2, hi2, facecolor="#62B1FF", edgecolor="0.5")
ax.fill_betweenx(press, low1, hi1, facecolor="#1873CC", edgecolor="#1873CC")
plt.semilogy(median, press, "b-", lw=1.5, label=r'$\rm Median$')
plt.semilogy(tb,     press,  "-", lw=1.5, label=r"$\rm Best\ fit$", color="r")
plt.ylim(press[0], press[-1])
ax.set_xticks([0, 600, 1200, 1800, 2400])
plt.xlim(0, 2000)
axl = plt.legend(loc="upper right", fontsize=fs-1)
axl.get_frame().set_alpha(0.6)
ax.set_yticklabels([""])
ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e-0, 1e2])
plt.xlabel(r"${\rm Temperature\ \ (K)}$", fontsize=fs)

nb = 14
lhx = 0.59
dhx = 0.12
dhy = 0.3
ax = plt.axes([lhx,            0.33 + dhy, dhx, dhy]) # TOP LEFT
ax.set_yticklabels([""])
h, hx, patch = plt.hist(rad, bins=nb)
ax.set_xticks([29000, 30000])
plt.xlabel(r"${\rm Radius\ (km)}$", fontsize=fs)
ax = plt.axes([lhx,            0.18,       dhx, dhy]) # BOT LEFT
ax.set_yticklabels([""])
h, hx, patch = plt.hist(poffset, bins=nb)
ax.set_xticks([0,100, 200])
plt.xlabel(r"${\rm WFC3\ offset\ (ppm)}$", fontsize=fs)
ax = plt.axes([lhx+0.02+  dhx, 0.33 + dhy, dhx, dhy]) # TOP CEN
ax.set_yticklabels([""])
h, hx, patch = plt.hist(pH2O, bins=nb)
ax.set_xticks([-3, -2, -1, 0])
plt.xlabel(r"$\log_{10}({\rm H2O})$", fontsize=fs)
ax = plt.axes([lhx+0.02+  dhx, 0.18,       dhx, dhy]) # BOT CEN
ax.set_yticklabels([""])
h, hx, patch = plt.hist(pCH4, bins=nb)
plt.xlim(hx[0], 0)
ax.set_xticks([-10, -5, 0])
plt.xlabel(r"$\log_{10}({\rm CH4})$", fontsize=fs)
ax = plt.axes([lhx+0.04+2*dhx, 0.33 + dhy, dhx, dhy]) # TOP RIGHT
ax.set_yticklabels([""])
h, hx, patch = plt.hist(pCO, bins=nb)
ax.set_xticks([-10, -5, 0])
plt.xlabel(r"$\log_{10}({\rm CO})$", fontsize=fs)
ax = plt.axes([lhx+0.04+2*dhx, 0.18,       dhx, dhy]) # BOT RIGHT
ax.set_yticklabels([""])
ax.set_xticks([-15, -10, -5, 0])
h, hx, patch = plt.hist(pCO2, bins=nb)
ax.set_xlim(hx[0], 0)
plt.xlabel(r"$\log_{10}({\rm CO2})$", fontsize=fs)

plt.savefig("plots/HAT-P-11b_atmosphere.ps")
