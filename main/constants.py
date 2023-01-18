from math import pi
from astropy import constants

# Setting constants
sigma   = constants.sigma_sb.value
c       = constants.c.value
a       = 4.*sigma/c
G       = constants.G.value
m_p     = constants.m_p.value
m_e     = constants.m_e.value
hbar    = constants.hbar.value
k       = constants.k_B.value


X       = 0.70                        # Fraction of Hydrogen
Z       = 0.02                        # Fraction of Metals
#X       = 0.71
#Z       = 0.02
Y       = 1 - X - Z                     # Fraction of Helium
mu      = (2*X+0.75*Y+0.5*Z)**(-1)      # mean molecular mass
gamma   = 5/3.
kappa_es = 0.02*(1+X) 

R_sun   = constants.R_sun.value
M_sun   = constants.M_sun.value
L_sun   = constants.L_sun.value

del(constants)