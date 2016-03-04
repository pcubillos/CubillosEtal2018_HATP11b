import numpy as np
import scipy.constants as sc

def Blamda(wavel, temp):
  """
  Planck function as function of wavelength in mks units.
  
  Parameters:
  -----------
  wavel: [Type] 1D ndarray, [Units] meters
         Wavelengths to sample the Planck function.
  temp: [type] Scalar, [Units] Kelvin degrees.
        Temperature of the blackbody.
        
  Returns:
  --------
  Blambda: [Type] 1D ndarray, [Units] W m^-2 sr^-2 m^-1
      The planck function for temperature temp evaluated at wavelengths wavel.
  """
  return  2 * sc.h * sc.c**2 / wavel**5 / (np.exp(sc.h*sc.c/(wavel*sc.k*temp)) - 1)

def Bnu(nu, temp):
  """
  Planck function as function of frequency in mks units.
 
  Parameters:
  -----------
  nu: [Type] 1D ndarray, [Units] Hertz.
        Frequencies to sample the Planck function.
  temp: [Type] Scalar, [Units] Kelvin degrees.
        Temperature of the blackbody.
        
  Returns:
  --------
  Bnu: [Type] 1D ndarray, [Units] W m^-2 sr^-2 Hz^-1
       The planck function for temperature temp evaluated at frequencies nu.
  """
  return 2 * sc.h * nu**3 / sc.c**2 / (np.exp(sc.h*nu/(sc.k*temp)) - 1)


def Bwn(wn, temp):
  """
  Planck function as function of wavenumber in mks units.
 
  Parameters:
  -----------
  nu: [Type] 1D ndarray, [Units] m-1.
        Frequencies to sample the Planck function.
  temp: [Type] Scalar, [Units] Kelvin degrees.
        Temperature of the blackbody.
        
  Returns:
  --------
  Bnu: [Type] 1D ndarray, [Units] W m^-2 sr^-2 m
       The planck function for temperature temp evaluated at wavenumber wn.
  """
  return 2 * sc.h * sc.c**2 * wn**3 / (np.exp(sc.h*sc.c*wn/(sc.k*temp)) - 1)


def Teq(Tstar, Rstar, a, A=0, f=1, Tsunc=0, Rsunc=0, aunc=0):
  """
  Calculates the equilibrium temperature of a planet and its uncertainty.

  Parameters:
  -----------
  Tstar: Scalar
         Host star temperature in kelvin.
  Rstar: Scalar
         Host star radius in meters.
  a: Scalar
     Orbital semi-major axis in meters.
  A: Scalar
     Planetary bond albedo.
  f: Scalar
     Planetary energy redistribution factor.
  
  Returns:
  --------
  teq: 1D ndarray
       Planet equilibrium temperature in kelvin deg.
  teqerr: 1D ndarray
          Equilibrium temperature uncertainty.
  
  """
  teq = ( (1.0-A)/(4.0*f) )**0.25 * (Rstar/a)**0.5 * Tstar
  tequnc = teq*np.sqrt( (Tsunc/Tstar)**2.0 + (aunc/(2*a))**2.0 +
                        (Rsunc/(2*Rstar))**2.0                 )

  return teq, tequnc

"""
#---------------------------------------------------------

rj = 7.149e07 # jupiter radius meters

#-----#  HD 217107  #-----#
d      = 6.085e17     # distance  meters

Star   = Blamda(lam*1e-6, Ts)
Planet = Blamda(lam*1e-6, Tp)

StarFlux    = Star  *np.pi*(Rstar /d)**2
PlanetFlux1 = Planet*np.pi*(1.0*rj/d)**2

FluxRatio = Planet/Star * ( (Rplan)/(Rstar) )**2.0

plt.figure(2)
plt.clf()

plt.plot([0.7, 0.7], [1e-20,1],'--g')
plt.plot([0.8, 0.8], [1e-20,1],'--g')
plt.plot([0.9, 0.9], [1e-20,1],'--g')
plt.semilogy(lam, FluxRatio, 'b', label=starname)


plt.xlim(0.3, 1.5)
plt.ylim(1e-8,1e-4)
plt.legend(loc='lower right')

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel('Flux ratios')
plt.savefig('/home/patricio/master/done/ps/thesis_fr' + filename)

#---------------------------------------------------------
"""
