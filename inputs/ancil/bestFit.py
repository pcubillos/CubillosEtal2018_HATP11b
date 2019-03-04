# Copyright (C) 2015-2016 University of Central Florida. All rights reserved.
# BART is under an open-source, reproducible-research license (see LICENSE).

"""
    This code runs and processes best-fit Transit run outputs.

    Functions
    ---------
    read_MCMC_out:
          Read the MCMC output log file. Extract the best fitting parameters.
    get_params:
	      Get correct number of all parameters from stepsize
    get_starData:
          Extract stellar temperature, radius, and mass from TEP file
    write_atmfile:
          Write best-fit atm file with scaled H2 and He to abundances sum of 1
    bestFit_tconfig:
          Write best-fit config file for best-fit Transit run
    callTransit:
          Call Transit to produce best-fit outputs. Plot MCMC posterior PT plot.
    plot_bestFit_Spectrum:
          Plot BART best-model spectrum
    plot_abun:
          Plot abundance profiles from best-fit run

    Revisions
    ---------
    2015-05-03  Jasmina  Original implementation
    2015-07-12  Jasmina  Added documentation.
    2018-12-21  Michael  Updated documentation, improved code readability.
    2019-02-13  Michael  Refactored callTransit. Enabled plotting for madhu 
                         PT profiles.
"""

import os, sys, subprocess
import numpy as np
import reader as rd
import scipy.constants as sc
import scipy.special   as sp
import scipy.interpolate as si
from scipy.ndimage.filters import gaussian_filter1d as gaussf
import matplotlib
import matplotlib.pyplot as plt

import makeatm as mat
import PT as pt
import cf
import wine as w
import readtransit as rt
import constants as c


def read_MCMC_out(MCfile):
    """
    Read the MCMC output log file. Extract the best fitting parameters.

    Parameters
    ----------
    MCfile: string. Path to MCMC output log file.

    Returns
    -------
    bestP: 1D array. Best-fit parameters.
    uncer: 1D array. Uncertainties for each parameter.
    """
    # Open file to read
    f = open(MCfile, 'r')
    lines = np.asarray(f.readlines())
    f.close()

    # Find where the data starts and ends:
    for ini in np.arange(len(lines)):
        if lines[ini].startswith(' Best-fit params'):
            break
    ini += 1
    end  = ini
    for end in np.arange(ini, len(lines)):
        if lines[end].strip() == "":
            break

    # Read data:
    bestP = np.zeros(end-ini, np.double)
    uncer = np.zeros(end-ini, np.double)
    for i in np.arange(ini, end):
        parvalues    = lines[i].split()
        bestP[i-ini] = parvalues[0]
        uncer[i-ini] = parvalues[1]

    return bestP, uncer


def get_params(bestP, stepsize, params):
    """
    Get correct number of all parameters from stepsize
    """
    j = 0
    allParams = np.zeros(len(stepsize))
    for i in np.arange(len(stepsize)):
        if stepsize[i] != 0.0:
            allParams[i] = bestP[j]
            j +=1
        else:
            allParams[i] = params[i]

    return allParams


def get_starData(tepfile):
    """
    Extract the Stellar temperature, radius, and mass from a TEP file.

    Parameters
    ----------
    tepfile: string. Path to Transiting ExoPlanet (TEP) file.

    Returns
    -------
    Rstar: float. Radius of the star in meters.
    Tstar: float. Temperature of the star in Kelvin.
    sma  : float. Semimajor axis in meters.
    gstar: float. Logarithm of the star's surface gravity in cgs units.
    """
    # Open tepfile to read and get data:
    tep = rd.File(tepfile)

    # Get star temperature in Kelvin:
    Tstar = np.float(tep.getvalue('Ts')[0])

    # Get star radius in MKS units:
    Rstar = np.float(tep.getvalue('Rs')[0]) * c.Rsun

    # Get semi major axis in meters:
    sma = np.float(tep.getvalue('a')[0]) * sc.au

    # Get star loggstar:
    gstar = np.float(tep.getvalue('loggstar')[0])

    return Rstar, Tstar, sma, gstar


def write_atmfile(atmfile, abun_file, molfit, T_line, abun_fact, date_dir,
                  p0, Rp, grav):
    """
    Write best-fit atm file with scaled H2 and He to abundances sum of 1.

    Parameters:
    -----------
    atmfile: String
      Atmospheric file to take the pressure and abundance profiles.
    molfit: 1D string ndarray
      List of molecule names to modify their abundances.
    rad: 1D float ndarray
      Modified radius of the atmospheric layers.
    T_line: 1D float ndarray
      Modified temperature of the atmospheric layers.
    abun_fact: 1D float ndarray
      List of scaling factors to modify the abundances of molfit molecules.
    date_dir: String
      Directory where to store the best-fit atmospheric file.
    """
    # Open atmfile to read
    f = open(atmfile, 'r')
    lines = np.asarray(f.readlines())
    f.close()

    # Get the molecules
    imol      = np.where(lines == "#SPECIES\n")[0][0] + 1
    molecules = lines[imol].split()

    # Find the line where the layers info begins
    start     = np.where(lines == "#TEADATA\n")[0][0] + 2
    headers   = lines[start-1].split()
    datalines = lines[start:]

    # Number of columns
    ncol = len(lines[start].split())

    # Number of layers
    ndata = len(datalines)

    # Allocate space for pressure:
    pressure = np.zeros(ndata, np.double)

    # Number of abundances (elements per line except Radius, Press and T)
    nabun = len(lines[start].split()) - 3

    data = np.zeros((ndata, len(headers)))
    for i in np.arange(ndata):
        data[i] = datalines[i].split()

    # Extract pressure data:
    pressure = data[:,1]

    # Fill out the abundances array:
    abundances = np.zeros((len(molecules), ndata))
    for i in np.arange(len(molecules)):
        for j in np.arange(ndata):
            abundances[i] = data[:, i+3]

    # recognize which columns to take from the atmospheric file
    headers = lines[start-1].split()
    columns = np.zeros(len(molfit), dtype=int)
    for i in np.arange(len(molfit)):
        for j in np.arange(len(headers)):
            if molfit[i] == headers[j]:
                columns[i] = j

    # number of molecules to fit:
    nfit = len(molfit)

    # multiply the abundances of molfit molecules
    for i in np.arange(len(columns)):
       abundances[columns[i]-3] *= 10**abun_fact[i]

    # ===== Scale H2 and He if sum abundances > 1 ===== #
    # Find index for Hydrogen and Helium
    molecules = np.asarray(molecules)
    iH2       = np.where(molecules=="H2")[0][0]
    iHe       = np.where(molecules=="He")[0][0]

    # Get H2/He abundance ratio:
    ratio = (abundances[iH2,:] / abundances[iHe,:])

    # Find which level has sum(abundances)>1 and get the difference
    q = np.sum(abundances, axis=0) - 1

    # Correct H2, and He abundances conserving their ratio:
    for i in np.arange(ndata):
    #    if q[i]>0:
            abundances[iH2, i] -= ratio[i] * q[i] / (1.0 + ratio[i])
            abundances[iHe, i] -=            q[i] / (1.0 + ratio[i])

    # Re-calculate the layers' radii using the Hydrostatic-equilibrium calc:
    # (Has to be in reversed order since the interpolation requires the
    #  pressure array in increasing order)
    mu = mat.mean_molar_mass(abun_file, spec=molecules, pressure=pressure,
                             temp=T_line, abundances=abundances.T)
    rad = mat.radpress(pressure[::-1], T_line[::-1], mu[::-1], p0, Rp, grav)
    rad = rad[::-1]

    # open best fit atmospheric file
    fout = open(date_dir + 'bestFit.atm', 'w')
    fout.writelines(lines[:start])

    # Write atm file for each run
    for i in np.arange(ndata):
        # Radius, pressure, and temp for the current line
        radi  = str('%10.3f'%rad[i])
        presi = str('%10.4e'%pressure[i])
        tempi = str('%7.2f'%T_line[i])

        # Insert radii array
        fout.write(radi.ljust(10) + ' ')

        # Insert results from the current line (T-P) to atm file
        fout.write(presi.ljust(10) + ' ')
        fout.write(tempi.ljust(7) + ' ')

        # Write current abundances
        for j in np.arange(len(headers) - 3):
            fout.write('%1.4e'%abundances[j][i] + ' ')
        fout.write('\n')

    # Close atm file
    fout.close()


def bestFit_tconfig(tconfig, date_dir, radius=None):
  '''
  Write best-fit config file for best-fit Transit run
  '''
  # Open atmfile to read
  f     = open(date_dir + tconfig, 'r')
  lines = np.asarray(f.readlines())
  f.close()

  for i in np.arange(len(lines)):
      # Change name to the atmfile in line zero
      if  lines[i].startswith("atm "):
          lines[i] = 'atm ' + date_dir + 'bestFit.atm' + '\n'
      # Change refradius:
      if  lines[i].startswith("refradius ") and radius is not None:
          lines[i] = 'refradius {}\n'.format(str(radius))

  # Write lines into the bestFit config file
  f = open(date_dir + 'bestFit_tconfig.cfg', 'w')
  f.writelines(lines)
  #f.writelines('savefiles yes')
  f.close()


def callTransit(atmfile, tepfile,  MCfile, stepsize,  molfit,  solution, p0, 
                tconfig, date_dir, burnin, abun_file, PTtype,  PTfunc, 
                filters, ctf=None):
    """
    Call Transit to produce best-fit outputs.
    Plot MCMC posterior PT plot.

    Parameters:
    -----------
    atmfile: String
       Atmospheric file.
    tepfile: String
       Transiting extra-solar planet file.
    MCfile: String
       File with MCMC log and best-fitting results.
    stepsize: 1D float ndarray
       Specified step sizes for each parameter.
    molfit: 1D String ndarray
       List of molecule names to modify their abundances.
    solution: String
       Flag to indicate transit or eclipse geometry
    p0: Float
       Atmosphere's 'surface' pressure level.
    tconfig: String
       Transit  configuration file.
    date_dir: String
       Directory where to store results.
    burnin: Integer
    abun_file: String
       Elemental abundances file.
    PTtype: String
       Pressure-temperature profile type ('line' for Line et al. 2013, 
       'madhu_noinv' or 'madhu_inv' for Madhusudhan & Seager 2009, 
       'iso' for isothermal)
    PTfunc: pointer to function
       Determines the method of evaluating the PT profile's temperature
    filters: list, strings.
       Filter files associated with the eclipse/transit depths
    ctf: 2D array.
       Contribution or transmittance functions corresponding to `filters`
    """
    # make sure burnin is an integer
    burnin = int(burnin)

    # read atmfile
    molecules, pressure, temp, abundances = mat.readatm(atmfile)
    # get surface gravity
    grav, Rp = mat.get_g(tepfile)
    # get star data if needed
    if PTtype == 'line':
      R_star, T_star, sma, gstar = get_starData(tepfile)
      # FINDME: Hardcoded value:
      T_int  = 100  # K
      PTargs = [R_star, T_star, T_int, sma, grav*1e2]
    else:
      PTargs = None # For non-Line profiles

    # Get best parameters
    bestP, uncer = read_MCMC_out(MCfile)
    allParams    = bestP

    # get PTparams and abundances factors
    nparams   = len(allParams)
    nmol      = len(molfit)
    nradfit   = int(solution == 'transit')
    # EDIT:
    noffset = 1
    nPTparams = nparams - nmol - nradfit - noffset

    PTparams  = allParams[:nPTparams]

    # call PT profile generator to calculate temperature
    best_T = pt.PT_generator(pressure, PTparams, PTfunc, PTargs)

    # Plot best PT profile
    plt.figure(1)
    plt.clf()
    plt.semilogy(best_T, pressure, '-', color = 'r')
    plt.xlim(0.9*min(best_T), 1.1*max(best_T))
    plt.ylim(max(pressure), min(pressure))
    plt.title('Best PT', fontsize=14)
    plt.xlabel('T (K)'     , fontsize=14)
    plt.ylabel('logP (bar)', fontsize=14)
    # Save plot to current directory
    plt.savefig(date_dir + 'Best_PT.png')

    # Update R0, if needed:
    if nradfit:
      Rp = allParams[nPTparams]

    # write best-fit atmospheric file
    # EDIT:
    write_atmfile(atmfile, abun_file, molfit, best_T,
                  allParams[nPTparams+nradfit+noffset:], date_dir, p0, Rp, grav)

    # write new bestFit Transit config
    if solution == 'transit':
      bestFit_tconfig(tconfig, date_dir, allParams[nPTparams])
    else:
      bestFit_tconfig(tconfig, date_dir)

    # Call Transit with the best-fit tconfig
    Transitdir      = os.path.dirname(os.path.realpath(__file__)) + \
                      "/../modules/transit/"
    bf_tconfig      = date_dir   + 'bestFit_tconfig.cfg'
    Tcall           = Transitdir + "/transit/transit"
    subprocess.call(["{:s} -c {:s}".format(Tcall, bf_tconfig)],
shell=True, cwd=date_dir)

    # ========== plot MCMC PT profiles ==========
    # get MCMC data:
    MCMCdata = date_dir + "/output.npy"
    data = np.load(MCMCdata)
    nchains, npars, niter = np.shape(data)

    # stuck chains:
    data_stack = data[0,:,burnin:]
    for c in np.arange(1, nchains):
        data_stack = np.hstack((data_stack, data[c, :, burnin:]))

    # create array of PT profiles
    PTprofiles = np.zeros((np.shape(data_stack)[1], len(pressure)))

    # current PT parameters for each chain, iteration
    curr_PTparams = PTparams

    # fill-in PT profiles array
    if ctf is None:
        print("  Plotting MCMC PT profile figure.")
    for k in np.arange(0, np.shape(data_stack)[1]):
        j = 0
        for i in np.arange(len(PTparams)):
            if stepsize[i] != 0.0:
                curr_PTparams[i] = data_stack[j,k]
                j +=1
            else:
                pass
        PTprofiles[k] = pt.PT_generator(pressure, curr_PTparams, 
                                        PTfunc, PTargs)

    # get percentiles (for 1,2-sigma boundaries):
    low1   = np.percentile(PTprofiles, 16.0, axis=0)
    hi1    = np.percentile(PTprofiles, 84.0, axis=0)
    low2   = np.percentile(PTprofiles,  2.5, axis=0)
    hi2    = np.percentile(PTprofiles, 97.5, axis=0)
    median = np.median(    PTprofiles,       axis=0)

    # plot figure
    plt.figure(2, dpi=300)
    plt.clf()
    if ctf is not None:
        plt.subplots_adjust(wspace=0.15)
        ax1=plt.subplot(121)
    else:
        ax1=plt.subplot(111)
    ax1.fill_betweenx(pressure, low2, hi2, facecolor="#62B1FF", 
                      edgecolor="0.5")
    ax1.fill_betweenx(pressure, low1, hi1, facecolor="#1873CC",
                      edgecolor="#1873CC")
    plt.semilogy(median, pressure, "-", lw=2, label='Median',color="k")
    plt.semilogy(best_T, pressure, "-", lw=2, label="Best fit", color="r")
    plt.ylim(pressure[0], pressure[-1])
    plt.legend(loc="best")
    plt.xlabel("Temperature  (K)", size=15)
    plt.ylabel("Pressure  (bar)",  size=15)
    if ctf is not None:
        # Add contribution or transmittance functions
        ax2=plt.subplot(122, sharey=ax1)
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(filters)))
        ax2.set_prop_cycle(plt.cycler('color', colormap))
        # Plot with filter labels
        for i in np.arange(len(filters)):
            (head, tail) = os.path.split(filters[i])
            lbl          = tail[:-4]
            ax2.semilogy(ctf[i], pressure, '--',  linewidth = 1, label=lbl)
        # Hide y axis tick labels
        plt.setp(ax2.get_yticklabels(), visible=False)
        # Place legend off figure in case there are many filters
        lgd = ax2.legend(loc='center left', bbox_to_anchor=(1,0.5), 
                         ncol=len(filters)//30 + 1, prop={'size':8})
        if solution == 'eclipse':
            ax2.set_xlabel('Normalized Contribution\nFunctions',  fontsize=15)
        else:
            ax2.set_xlabel('Transmittance', fontsize=15)
    

    # save figure
    if ctf is not None:
        savefile = date_dir + "MCMC_PTprofiles_cf.png"
        plt.savefig(savefile, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        savefile = date_dir + "MCMC_PTprofiles.png"
        plt.savefig(savefile)


# EDIT:
def plot_bestFit_Spectrum(filters, kurucz, tepfile, solution, output, data,
                          uncert, date_dir, offset=0.0):
    '''
    Plot BART best-model spectrum

    Parameters
    ----------
    filters : list, strings. Paths to filter files corresponding to data.
    kurucz  : string. Path to Kurucz stellar model file.
    tepfile : string. Path to Transiting ExoPlanet (TEP) file.
    solution: string. Observing geometry. 'eclipse' or 'transit'.
    output  : string. Best-fit spectrum output file name.
    data    : 1D array. Eclipse or transit depths.
    uncert  : 1D array. Uncertainties for data values.
    date_dir: string. Path to directory where the plot will be saved.
    '''
    # get star data
    R_star, T_star, sma, gstar = get_starData(tepfile)

    # get surface gravity
    grav, Rp = mat.get_g(tepfile)

    # convert Rp to m
    Rp = Rp * 1000

    # ratio planet to star
    rprs = Rp/R_star

    # read kurucz file
    starfl, starwn, tmodel, gmodel = w.readkurucz(kurucz, T_star, gstar)

    # read best-fit spectrum output file, take wn and spectra values
    if solution == 'eclipse':
        specwn, bestspectrum = rt.readspectrum(date_dir + output, wn=True)
        # print on screen
        print("  Plotting BART best-fit eclipse spectrum figure.")
    elif solution == 'transit':
        specwn, bestspectrum = rt.readspectrum(date_dir + output, wn=True)
        # print on screen
        print("  Plotting BART best-fit modulation spectrum figure.")

    # convert wn to wl
    specwl = 1e4/specwn

    # number of filters
    nfilters = len(filters)

    # read and resample the filters:
    nifilter  = [] # Normalized interpolated filter
    istarfl   = [] # interpolated stellar flux
    wnindices = [] # wavenumber indices used in interpolation
    meanwn    = [] # Filter mean wavenumber
    for i in np.arange(nfilters):
        # read filter:
        filtwaven, filttransm = w.readfilter(filters[i])
        meanwn.append(np.sum(filtwaven*filttransm)/sum(filttransm))
        # resample filter and stellar spectrum:
        nifilt, strfl, wnind = w.resample(specwn, filtwaven, filttransm,
                                          starwn, starfl               )
        nifilter .append(nifilt)
        istarfl  .append(strfl)
        wnindices.append(wnind)

    # convert mean wn to mean wl
    meanwl = 1e4/np.asarray(meanwn)

    # band-integrate the flux-ratio or modulation:
    bandflux = np.zeros(nfilters, dtype='d')
    bandmod  = np.zeros(nfilters, dtype='d')
    for i in np.arange(nfilters):
        fluxrat = (bestspectrum[wnindices[i]]/istarfl[i]) * rprs*rprs
        bandflux[i] = w.bandintegrate(fluxrat, specwn, nifilter[i],
                                                                 wnindices[i])
        bandmod[i]  = w.bandintegrate(bestspectrum[wnindices[i]],
                                            specwn, nifilter[i], wnindices[i])

    # stellar spectrum on specwn:
    sinterp = si.interp1d(starwn, starfl)
    sflux   = sinterp(specwn)
    frat    = bestspectrum/sflux * rprs * rprs

    # EDIT:
    nWFC3  = 29    # Number of WFC3 points
    data[0:nWFC3] -= offset

    # plot figure
    plt.rcParams["mathtext.default"] = 'rm'
    matplotlib.rcParams.update({'mathtext.default':'rm'})
    matplotlib.rcParams.update({'font.size':10})
    plt.figure(3, (8.5, 5))
    plt.clf()

    # depending on solution plot eclipse or modulation spectrum
    if solution == 'eclipse':
        gfrat = gaussf(frat, 2)
        plt.semilogx(specwl, gfrat*1e3, "b", lw=1.5, label="Best-fit")
        plt.errorbar(meanwl, data*1e3, uncert*1e3, fmt="or", label="data")
        plt.plot(meanwl, bandflux*1e3, "ok", label="model", alpha=1.0)
        plt.ylabel(r"$F_p/F_s$ (10$^{-3}$)", fontsize=12)

    elif solution == 'transit':
        gmodel = gaussf(bestspectrum, 2)
        plt.semilogx(specwl, gmodel, "b", lw=1.5, label="Best-fit")
        plt.errorbar(meanwl, data, uncert, fmt="or", label="data")
        plt.plot(meanwl, bandmod, "ok", label="model", alpha=0.5)
        plt.ylabel(r"$(R_p/R_s)^2$", fontsize=12)

    leg = plt.legend(loc="lower right")
    leg.get_frame().set_alpha(0.5)
    ax = plt.subplot(111)
    ax.set_xscale('log')
    plt.xlabel(ur"${\rm Wavelength\ \ (\u03bcm)}$", fontsize=12)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if   max(specwl) - min(specwl) > 30:
        ax.set_xticks(np.arange(round(min(specwl)),max(specwl),4))
    elif max(specwl) - min(specwl) > 15:
        ax.set_xticks(np.arange(round(min(specwl)),max(specwl),2))
    else:
        ax.set_xticks(np.arange(round(min(specwl)),max(specwl),1))
    plt.xlim(min(specwl),max(specwl))
    plt.savefig(date_dir + "BART-bestFit-Spectrum.png")


def plotabun(date_dir, atmfile, molfit):
    '''
    Plot abundance profiles from best fit run.

    Input
    -----
    date_dir: string
      Path to BART output directory

    atmfile: string
      Name of best fit atmospheric file

    molfit:  1D string array
      Molecules to plot
    '''
    # Import best fit atmosphere results
    species, pressure, temp, abundances = mat.readatm(date_dir + atmfile)

    # Create array of indices for species to plot
    molfitindex = np.zeros(len(molfit), dtype='int')

    k = 0
    # Find the index of each species within the atmosphere file
    for i in range(len(species)):
        for j in range(len(molfit)):
            if species[i] == molfit[j]:
                molfitindex[k] = i
                k += 1

    plt.clf()

    # Plot the abundance profile of each species
    for i in range(len(molfit)):
        plt.loglog(abundances[:,molfitindex[i]], pressure,
                   label=species[molfitindex[i]], linewidth=4)

    plt.legend(loc='upper left')
    plt.xlabel('Molar Mixing Fraction')
    plt.ylabel('Pressure (bars)')
    plt.title('Best Fit Abundance Profiles')
    plt.gca().invert_yaxis()
    plt.savefig(date_dir + 'abun_profiles.png')

