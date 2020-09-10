#! /usr/bin/env python

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gaussf

sys.path.append("BART/modules/transit/scripts")
import readtransit as rt
sys.path.append("BART/code")
from constants import Rsun, Rjup
sys.path.append("inputs/ancil")
import blackbody as bb

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Read Transit spectra:
wl, flux_H2O = rt.readspectrum("run02_H2O/emission_H2O-pands_1-20um.dat", 0)
wl, flux_CO  = rt.readspectrum("run03_CO/emission_CO-HITEMP_1-20um.dat",  0)
wl, flux_CO2 = rt.readspectrum("run04_CO2/emission_CO2-HITEMP_1-20um.dat", 0)
wl, flux_CH4 = rt.readspectrum("run05_CH4/emission_CH4-ExoMol_1-20um.dat", 0)

# Read Morley's CIA spectrum:
mfiles = ["inputs/Morley/h2he.spec",    "inputs/Morley/h2o_h2he.spec",
          "inputs/Morley/co_h2he.spec", "inputs/Morley/co2_h2he.spec",
          "inputs/Morley/ch4_h2he.spec"]

mdata = []
for j in np.arange(len(mfiles)):
  f = open(mfiles[j], "r")
  lines = f.readlines()
  f.close()

  nwl = len(lines)-2
  mflux = np.zeros(nwl)
  wl_mor  = np.zeros(nwl)

  for i in np.arange(nwl):
    data = lines[i+2].split()
    wl_mor[i] = data[0]
    mflux[i]  = data[1]

  # Convert from F_wavelength to F_wavenumber, and from MKS to CGS:
  mflux *= (1e-6*wl_mor)**2.0 * 1e5

  mdata.append(mflux)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Calculate flux ratios:

# Wavenumber in m-1:
wn = 1e6/wl
wn_mor = 1e6/wl_mor
tstar = 5700.0
# Blackbody spectra:
Fstar     = np.pi * bb.Bwn(wn,     tstar) * 1e5
Fstar_mor = np.pi * bb.Bwn(wn_mor, tstar) * 1e5
Ftop      = np.pi * bb.Bwn(wn,   1089.33) * 1e5
Fbot      = np.pi * bb.Bwn(wn,   1619.32) * 1e5

sigma = 1.25

tfr_H2O = gaussf(flux_H2O/Fstar *     (Rjup/Rsun)**2*1e3, sigma)
mfr_H2O = gaussf(mdata[1]/Fstar_mor * (Rjup/Rsun)**2*1e3, sigma)

tfr_CO  = gaussf(flux_CO /Fstar *     (Rjup/Rsun)**2*1e3, sigma)
mfr_CO  = gaussf(mdata[2]/Fstar_mor * (Rjup/Rsun)**2*1e3, sigma)

tfr_CO2 = gaussf(flux_CO2/Fstar *     (Rjup/Rsun)**2*1e3, sigma)
mfr_CO2 = gaussf(mdata[3]/Fstar_mor * (Rjup/Rsun)**2*1e3, sigma)

tfr_CH4 = gaussf(flux_CH4/Fstar *     (Rjup/Rsun)**2*1e3, sigma)
mfr_CH4 = gaussf(mdata[4]/Fstar_mor * (Rjup/Rsun)**2*1e3, sigma)


fr_top = Ftop/Fstar * (Rjup/Rsun)**2*1e3
fr_bot = Fbot/Fstar * (Rjup/Rsun)**2*1e3

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Make plot:
color = ["darkorange", "b"]
alpha = 1.0
lw = 1.5
xran = 1, 20
yran = 0, 2.3
fs = 18
matplotlib.rc('xtick', labelsize=fs-3)
matplotlib.rc('ytick', labelsize=fs-3)

plt.figure(-22, (7.5, 10))
plt.clf()
plt.subplots_adjust(0.12, 0.07, 0.98, 0.99, hspace=0.0,wspace=0.0)
ax = plt.subplot(411)  # H2O
plt.semilogx(wl, fr_top, "--", color="0.5", lw=lw)
plt.semilogx(wl, fr_bot, "--", color="0.5", lw=lw)
plt.semilogx(wl_mor, mfr_H2O, color[0], lw=lw, alpha=1.0,
    label=r"Morley")
plt.semilogx(wl, tfr_H2O, color[1], lw=lw, alpha=alpha,
    label=r"Transit code")
ax.set_xticklabels([])
plt.xlim(xran)
plt.ylim(yran)
plt.text(15, 0.2, r"H$_2$O", fontsize=fs)
plt.legend(loc="upper left", fontsize=fs-2)
plt.ylabel(r"$F_{\rm p}/F_{\rm s}$  (ppt)", fontsize=fs)
ax = plt.subplot(412)  # CO
plt.semilogx(wl, fr_top, "--", color="0.5", lw=lw)
plt.semilogx(wl, fr_bot, "--", color="0.5", lw=lw)
plt.semilogx(wl_mor, mfr_CO,  color[0], lw=lw, alpha=1.0)
plt.semilogx(wl, tfr_CO,  color[1], lw=lw, alpha=alpha)
ax.set_xticklabels([])
plt.xlim(xran)
plt.ylim(yran)
plt.text(15, 0.2, r"${\rm CO}$", fontsize=fs)
plt.ylabel(r"$F_{\rm p}/F_{\rm s}$  (ppt)", fontsize=fs)
ax = plt.subplot(413)  # CO2
plt.semilogx(wl, fr_top, "--", color="0.5", lw=lw)
plt.semilogx(wl, fr_bot, "--", color="0.5", lw=lw)
plt.semilogx(wl_mor, mfr_CO2, color[0], lw=lw, alpha=1.0)
plt.semilogx(wl, tfr_CO2, color[1], lw=lw, alpha=alpha)
ax.set_xticklabels([])
plt.xlim(xran)
plt.ylim(yran)
plt.text(15, 0.2, r"CO$_2$", fontsize=fs)
plt.ylabel(r"$F_{\rm p}/F_{\rm s}$  (ppt)", fontsize=fs)
ax = plt.subplot(414)  # CH4
plt.semilogx(wl, fr_top, "--", color="0.5", lw=lw)
plt.semilogx(wl, fr_bot, "--", color="0.5", lw=lw)
plt.semilogx(wl_mor, mfr_CH4,  color[0], lw=lw, alpha=1.0)
plt.semilogx(wl, tfr_CH4,  color[1], lw=lw, alpha=alpha)
plt.xlim(xran)
plt.ylim(yran)
plt.text(15, 0.2, r"CH$_4$", fontsize=fs)
plt.ylabel(r"$F_{\rm p}/F_{\rm s}$  (ppt)", fontsize=fs)
plt.xlabel(r"Wavelength (um)", fontsize=fs)

ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([1, 2, 3, 5, 10, 20])
plt.savefig("plots/emission_H2O-CO-CO2-CH4.pdf")
