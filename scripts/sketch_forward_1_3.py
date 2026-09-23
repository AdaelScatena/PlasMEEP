import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros
from scipy.optimize import brentq
from plasmeep.lib import Plasmeep as pm, WP
import meep as mp

# ***Annulus build***

#length scale
a = 0.01 # meters
res = 20 # pixels per a (development resolution)
dpml = 5.0 # PML thickness in units of a

#geometry units of a
R_p = 1.5 #plasma radius
r_in = 2.0 #scaffold inner radius
r_out = 5.0 #scaffold outer radius
eps_scaffold = 4.0 #placeholder uniform dielectricc - design variable

#domain size of a
nx = 28
ny = 28

#probe band in MEEP frequency units
f_min = 0.1
f_max = 0.3

print('a =', a, 'm')
print('f=1 in MEEP <->', 3e8/a/1e9, 'GHz')
print('probe band:', f_min*30, 'to', f_max*30, 'GHz')

#create empty simulation 'box'
model = pm(a, res, dpml, nx, ny)

print('cell size (in units of a):', model.cell)
print('PML thickness', dpml)

# Circular scaffold annulus
model.geometry.append(
    mp.Cylinder(
        radius=r_out,
        material=model.Get_Med(eps_scaffold),
        center=mp.Vector3(0, 0, 0),
    )
)
model.geometry.append(
    mp.Cylinder(
        radius=r_in,
        material=model.Get_Med(1.0),  # vacuum hole
        center=mp.Vector3(0, 0, 0),
    )
)

#  ***Port build***

#place 5 waveguide feeds around the scaffold
n_ports = 5
w = 0.8 #waveguide width
l = 3.0 #waveguide length
R_port = r_out+1.5+1/2
eps_wg = 4.0 #dielectric feed

port_centers = []
for k in range(n_ports):
    theta = k*2*np.pi/n_ports #0, 72, 144, 216, 288 degrees
    cx = R_port*np.cos(theta)
    cy = R_port*np.sin(theta)
    port_centers.append((cx, cy, theta))         

    # Unit vectors: e1 along the waveguide (radial), e2 across it
    e1 = mp.Vector3(np.cos(theta), np.sin(theta), 0)   # length direction
    e2 = mp.Vector3(-np.sin(theta), np.cos(theta), 0)  # width direction
    medium = model.Get_Med(eps_wg)
    model.geometry.append(
        mp.Block(
            size=mp.Vector3(l, w, mp.inf),
            center=mp.Vector3(cx, cy, 0),
            e1=e1,
            e2=e2,
            material=medium,
        )
    )

#port check via geometry print
print('port angles (deg):', [p[2] *180/np.pi for p in port_centers])
print('Source ports for later: k=0 and k=2 (144 deg apart)')

#  ***Plasma column build***

r = np.linspace(0,R_p,200)
BETA_SCHOTTKY = jn_zeros(0,1)[0]  

#converting functions
def bessel_profile(r, n0, beta, R_p):
    """n_e(r) = n0*J0(beta*r/R_p)."""
    return n0*j0(beta*r/R_p)

def h_from_beta(beta):
    return float(j0(beta))

def beta_from_h(h):
    #h = J0(beta), beta in (o,2.405]
    return (brentq(lambda b: j0(b)-h, 1e-6, BETA_SCHOTTKY))

#Cocentric drude shells for example profile 
N_shells = 12
n0_si = 5e16
h_ex = 0.4
beta_ex = beta_from_h(h_ex)

gamma_Hz=1e9
gamma_meep=model.Nondimensionalize_Freq(gamma_Hz)

shell_r=[]
shell_n=[]

for i in range(N_shells, 0, -1):
    r_out_shell=R_p*i/N_shells
    r_mid_shell=R_p*(i-0.5)/N_shells

    n_i = float(bessel_profile(r_mid_shell, n0_si, beta_ex, R_p))
    n_i = max(n_i, 1e10)

    fp_Hz = WP(n_i)/(2*np.pi)
    wp_meep = model.Nondimensionalize_Freq(fp_Hz)

    medium = model.Get_Med(1.0, wp=wp_meep, gamma=gamma_meep)
    model.geometry.append(
        mp.Cylinder(
            radius=r_out_shell,
            material=medium,
            center=mp.Vector3(0,0,0),
        )
    )

    shell_r.append(r_mid_shell)
    shell_n.append(n_i)
    print(f"shell {i:2d}: r_mid={r_mid_shell:.3f}, n={n_i:.3e} m^-3, fp={fp_Hz/1e9:.2f} GHz")


#  ***Add source to 1 horn***

cx0, cy0, theta0 = port_centers[0] #define source horn index

#show simulation with marked source horn
sim = model.Get_Sim()
sim.plot2D()
plt.plot(cx0, cy0, 'o', color='red', markersize=10, label='source port') #mark which port is the source port
plt.legend()
plt.title('Phase 1.3: scaffold + Drude shell')
plt.savefig('geometry_1_3_part1.png', dpi=200, bbox_inches='tight')

nfreq = 21 #number of frequencies
frequencies = np.linspace(f_min, f_max, nfreq) #array of frequencies
fcen = 0.5*(f_min + f_max) #center frequency
df = f_max-f_min #change in frequency from minimum to maximum

#define meep eigenmode source at indexed horn center
sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        center=mp.Vector3(cx0, cy0, 0),
        size=mp.Vector3(0, w, 0),
        direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(-np.cos(theta0), -np.sin(theta0)),
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_match_freq=True,
    )
]

eig_kpoint = mp.Vector3(-np.cos(theta0), -np.sin(theta0))

model.sources = sources
sim = model.Get_Sim()

mode_monitors = []
for (cx, cy, theta) in port_centers:
    m = sim.add_mode_monitor(
        fcen, df, nfreq,
        mp.ModeRegion(center=mp.Vector3(cx, cy, 0), size=mp.Vector3(0, w, 0)),
    )
    mode_monitors.append(m)
sim.run(until_after_sources=20)   # once, after the loop
coeffs = []
for m in mode_monitors:
    alpha = sim.get_eigenmode_coefficients(m, [1]).alpha
    coeffs.append(alpha)


coeffs = np.array(coeffs)
print('coeff shape:', coeffs.shape)
print('port 0 |c| vs freq (dir 0):', np.abs(coeffs[0, 0, :, 0]))
print('port 0 |c| vs freq (dir 1):', np.abs(coeffs[0, 0, :, 1]))
print('all ports, mid freq, both dirs:\n', np.abs(coeffs[:, 0, nfreq//2, :]))

    