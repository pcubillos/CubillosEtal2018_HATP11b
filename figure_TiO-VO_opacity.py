#! /usr/bin/env python
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants   as sc

f = open("run06_TiO-VO/opacity_TiO-VO_2200-2300K_0.1-1.0bar.dat", "rb")

# Read the number of values per array:
nmol  = struct.unpack('l', f.read(8))[0]
ntemp = struct.unpack('l', f.read(8))[0]
nrad  = struct.unpack('l', f.read(8))[0]
nwave = struct.unpack('l', f.read(8))[0]

# Read arrays:
mol   = np.asarray(struct.unpack(str(nmol )+'i', f.read(4*nmol )))
temp  = np.asarray(struct.unpack(str(ntemp)+'d', f.read(8*ntemp)))
press = np.asarray(struct.unpack(str(nrad) +'d', f.read(8*nrad)))
wn    = np.asarray(struct.unpack(str(nwave)+'d', f.read(8*nwave)))

# Read data:
ndata = nmol*ntemp*nrad*nwave
data = np.asarray(struct.unpack('d'*ndata, f.read(8*ndata)))
data = np.reshape(data, (nrad, ntemp, nmol, nwave))

# Get the wavelength (in microns):
wl = 1e4/np.asarray(wn)

# gr/molecule:
mass   = np.array([63.8664, 66.9409]) / sc.Avogadro

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Make plot:
lw = 1.0
fs = 14
matplotlib.rc('xtick', labelsize=fs-4)
matplotlib.rc('ytick', labelsize=fs-4)
color  = ["green", "darkorange"]
legend = ["TiO", "VO"]

r = 0  # Index for 10 bar layer
t = 0  # Index for 2200 K

plt.figure(2, (7.5, 4.5))
plt.clf()
plt.subplots_adjust(0.12, 0.12, 0.99, 0.95)
ax = plt.subplot(111)
plt.semilogy(wl, data[r,t,1]*mass[1], color="darkorange", label=r"${\rm VO}$")
plt.semilogy(wl, data[r,t,0]*mass[0], color="green",      label=r"${\rm TiO}$")
plt.xlim(0.4, 1.5)
plt.ylim(1e-22, 1e-15)
ax.tick_params(labelsize=fs-2)
plt.legend(loc="upper right", fontsize=fs)
plt.xlabel(r"Wavelength  (um)", fontsize=fs)
plt.ylabel(r"Opacity  (cm$^{2}$ molecule$^{-1}$)", fontsize=fs)
plt.savefig("plots/TiO-VO_opacity_10bar_2200K.pdf")

