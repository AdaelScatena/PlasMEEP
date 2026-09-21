#Defines geometry constraints

from plasmeep.lib import Plasmeep as pm
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

#length scale
a = 0.01 # meters
res = 20 # pixels per a (development resolution)
dpml = 5.0 # PML thickness in units of a

#domain size of a
nx = 24
ny = 24

#geometry units of a
R_p = 1.5 #plasma radius
r_in = 2.0 #scaffold inner radius
r_out = 5.0 #scaffold outer radius
eps_scaffold = 4.0 #placeholder uniform dielectricc - design variable

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

#square-annulus scaffold approx using blocks
half_out = r_out
model.Add_Block(
    low_left = [-half_out, -half_out, 0],
    size = [2*half_out, 2*half_out, 0],
    eps = eps_scaffold,
)
#cut out inner square by overlaying vacuum (eps=1)
half_in = r_in
model.Add_Block(
    low_left = [-half_in, -half_in, 0],
    size = [2*half_in, 2*half_in, 0],
    eps = 1.0,
)

#add plasma column at the origin (uniform vacuum for now, no Drude yet)
model.Add_Rod(R_p, [0,0,0], eps = 1.0)

#place 5 waveguide feeds around the scaffold
n_ports = 5
R_port = r_out+0.8 #radial distance to waveguide center
w = 0.8 #waveguide width
l = 0.8 #waveguide length
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

print('port angles (deg):', [p[2] *180/np.pi for p in port_centers])
print('Source ports for later: k=0 and k=2 (144 deg apart)')

#build a MEEP simulation object from model.geometry
sim = model.Get_Sim()

#plot permittivity 
sim.plot2D()
plt.title('Phase 1.1 sketch: plasma + scaffold placeholders')
plt.savefig("geometry_1_1_step3.png", dpi=200, bbox_inches="tight")
print('Complete')