import numpy as np
import scipy.constants as sc


mearth = 5.9724e+27
rearth = 637810000.0

# Solar H2/He ratio:
molecs = "He H2 H2O CH4 CO CO2".split()
Q = np.array([0.14551, 0.85364, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10])
nmol = len(molecs)

# Pressure profile:
nlayers = 101
pbottom = 1e+2
ptop = 1e-8
press = np.logspace(np.log10(pbottom), np.log10(ptop), nlayers)

# Isothermal temperature:
temp = 900.0

# Hydrostatic-equilibrium altitude:
radius = np.zeros(nlayers, np.double)
mu = 2.3 / sc.N_A
mplanet = 23.4 * mearth
rplanet = 4.36 * rearth
g = 1e3*sc.G * mplanet / rplanet**2
# Scale height in km:
H = 1e7*sc.k * temp / (mu*g) * 1e-5
# Radius at pbottom in km:
radius0 = 28312.643
radius = -H * np.log(press/pbottom) + radius0

# Write atmfile with unifrom abundances:
with open("run07_HAT-P-11b_BART_101layers/atmosphere_HAT-P-11b_uniform.atm", 'w') as f:
    f.write(
        '# TEA atmospheric file with uniform abundances\n\n'
        '# Units:\n'
        'ur 1e5\n'
        'up 1e6\n'
        'q  number\n\n'
        '#SPECIES\n')
    f.write(' '.join(molecs) + '\n\n')
    f.write('#TEADATA\n')
    f.write('#Radius    Pressure   Temp    ')
    f.write("".join(f'{molec:12s}' for molec in molecs))
    f.write('\n')
    for i in range(nlayers):
        f.write('{:10.3f} {:.4e} {:6.2f}'.format(radius[i], press[i], temp))
        for q in Q:
            f.write('  {:.4e}'.format(q))
        f.write('\n')


