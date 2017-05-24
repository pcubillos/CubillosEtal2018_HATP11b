#! /usr/bin/env python

import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants   as sc
from scipy.ndimage.filters import gaussian_filter1d as gaussf

sys.path.append("BART/modules/transit/scripts")
import readtransit as rt
sys.path.append("BART/code")
import makeatm as ma

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Read Transit CIA spectra:
wl, CIA_hit = rt.readspectrum("run01_CIA/emission_H2He_1.0-20um_HITRAN.dat", 0)
wl, CIA_bor = rt.readspectrum("run01_CIA/emission_H2He_1.0-20um_Borysow.dat", 0)

# Read Morley's CIA spectrum:
f = open("Morley/h2he.spec", "r")
lines = f.readlines()
f.close()

nwl = len(lines)-2
wl_mor  = np.zeros(nwl)
CIA_mor = np.zeros(nwl)

for i in np.arange(nwl):
  data = lines[i+2].split()
  wl_mor [i] = data[0]
  CIA_mor[i] = data[1]

# Convert from F_wavelength to F_wavenumber, and from MKS to CGS:
CIA_mor *= (1e-6*wl_mor)**2.0 * 1e5

# Read atmospheric model:
mols, press, temp, ab = ma.readatm("inputs/atmfile/uniform01.atm")


# Plot:
lw = 1.5
xran = 1, 20
yran = 0, 2.3
fs = 18
matplotlib.rc('xtick', labelsize=fs-4)
matplotlib.rc('ytick', labelsize=fs-4)

# Emission spectra side-by-side:
plt.figure(1, (8,4.5))
plt.clf()
plt.subplots_adjust(0.15, 0.12, 0.9, 0.95)
ax = plt.subplot(111)
#plt.plot(wl_mor, CIA_mor, "darkorange", lw=lw, label="Morley")
plt.plot(wl_mor, CIA_mor, "darkorange", lw=3.5, label=r"${\rm Morley}$")
plt.plot(wl,     CIA_hit, "b",          lw=lw, label=r"${\rm Transit/HITRAN}$")
plt.plot(wl,     CIA_bor, "limegreen",  lw=lw, label=r"${\rm Transit/Borysow}$")
plt.legend(loc="lower left")
plt.xlim(1, 20)
plt.ylim(0, 65000)
plt.xlabel(r"${\rm Wavelength\ \ (um)}$", fontsize=fs)
plt.ylabel(r"${\rm Day-side\ flux\ \ (ergs\ s}^{-1}{\rm cm}^{-1})$",
           fontsize=fs)
ax = plt.axes([0.67, 0.43, 0.20, 0.48])
plt.semilogy(temp, press, "r", lw=2)
plt.ylim(press[0], press[-1])
ax.set_xticks([1000, 1300, 1600])
ax.set_yticks([1e2, 1, 1e-2, 1e-4])
plt.xlabel(r"${\rm Temperature\ \ (K)}$", fontsize=fs-1)
plt.ylabel(r"${\rm Pressure\ \ (bar)}$",  fontsize=fs-1)
plt.savefig("plots/CIA_emission_spectra_1-20um.ps")
