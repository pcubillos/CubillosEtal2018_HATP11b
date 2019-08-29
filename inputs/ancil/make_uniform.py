import numpy as np


# Read TEA file:
atmfile = "./run07_HAT-P-11b_BART/BARTinputs/atmosphe_HAT-P-11b.atm"
with open(atmfile, 'r') as f:
    lines = f.readlines()

imol = lines.index("#SPECIES\n") + 1
start = lines.index("#TEADATA\n") + 2

molecs = lines[imol].split()

nlayers = len(lines) - start
nmol = len(molecs)

radius = np.zeros(nlayers, np.double)
press  = np.zeros(nlayers, np.double)
temp   = np.zeros(nlayers, np.double)
abund  = np.zeros((nlayers, nmol), np.double)
for i in np.arange(nlayers):
    line = lines[start+i].strip().split()
    radius[i] = line[0]
    press[i]  = line[1]
    temp[i]   = line[2]
    abund[i]  = line[3:]

# Set uniform profiles:
for molec in ['CO', 'CO2', 'CH4', 'H2O']:
    abund[:,molecs.index(molec)] = 1e-10

# Write atmfile with unifrom abundances:
uatmfile = "./run07_HAT-P-11b_BART/BARTinput/atmosphe_HAT-P-11b_uniforms.atm"
with open(uatmfile, 'w') as f:
    f.write('# TEA atmospheric file with uniform abundances\n\n'
            '# Units:\n'
            'ur 1e5\n'
            'up 1e6\n'
            'q  number\n\n'
            '#SPECIES\n')
    f.write(' '.join(molecs) + '\n\n')
    f.write('#TEADATA\n')
    f.write('#Radius    Pressure   Temp    Abundances\n')
    for i in range(nlayers):
        f.write('{:10.3f} {:.4e} {:6.2f}'.format(radius[i], press[i], temp[i]))
        for q in abund[i]:
            f.write('  {:.4e}'.format(q))
        f.write('\n')


