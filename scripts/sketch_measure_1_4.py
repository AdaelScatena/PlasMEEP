import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros
from scipy.optimize import brentq
from plasmeep.lib import Plasmeep as pm, WP
import meep as mp
from scipy.interpolate import RegularGridInterpolator

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

#print('a =', a, 'm')
#print('f=1 in MEEP <->', 3e8/a/1e9, 'GHz')
#print('probe band:', f_min*30, 'to', f_max*30, 'GHz')

#create empty simulation 'box'
model = pm(a, res, dpml, nx, ny)

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
    #print(f"shell {i:2d}: r_mid={r_mid_shell:.3f}, n={n_i:.3e} m^-3, fp={fp_Hz/1e9:.2f} GHz")

# ***Source and receiver horns***

# Horn 0 launches inward. All five horns, including that one, record the mode.
k_src = 0
nfreq = 21
fcen = 0.5*(f_min + f_max)
df = f_max - f_min
span = 4.0 #cut across the horn

def horn_cut(theta):
    """Axis-aligned line across a horn, and the outward radial k-point.

    Meep monitor lines lie on the grid axes. A horn that runs mostly along x
    is cut along y; a horn that runs mostly along y is cut along x. The mode
    still propagates along the horn, set by this k-point with NO_DIRECTION.
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    if abs(c) >= abs(s):
        size = mp.Vector3(0, span, 0)
    else:
        size = mp.Vector3(span, 0, 0)
    k_out = mp.Vector3(c, s, 0)
    return size, k_out

cx0, cy0, theta0 = port_centers[k_src]
r_src = R_port + 1.0 #outboard of the receiver line, still on the horn
sx = r_src*np.cos(theta0)
sy = r_src*np.sin(theta0)
size_src, k_out_src = horn_cut(theta0)

model.sources = [mp.EigenModeSource(
    src=mp.GaussianSource(frequency=fcen, fwidth=df),
    center=mp.Vector3(sx, sy, 0),
    size=size_src,
    direction=mp.NO_DIRECTION,
    eig_kpoint=mp.Vector3(-k_out_src.x, -k_out_src.y, 0),
    eig_band=1,
    eig_parity=mp.ODD_Z,
    eig_match_freq=True,
)]

sim = model.Get_Sim()

receivers = []
for k, (cx, cy, theta) in enumerate(port_centers):
    size_k, k_out = horn_cut(theta)
    mon = sim.add_mode_monitor(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(cx, cy, 0), size=size_k),
    )
    receivers.append(dict(
        k=k,
        monitor=mon,
        k_out=k_out,
        center=(cx, cy),
        size=size_k,
    ))
    #print(f"horn {k}: theta={theta*180/np.pi:.1f} deg, receiver")

#print(f"source horn k={k_src} at {theta0*180/np.pi:.1f} deg, launching inward")

sim.plot2D()
for rec in receivers:
    (x0, y0), (x1, y1) = (
        (rec["center"][0] - rec["size"].x/2, rec["center"][1] - rec["size"].y/2),
        (rec["center"][0] + rec["size"].x/2, rec["center"][1] + rec["size"].y/2),
    )
    plt.plot([x0, x1], [y0, y1], color="dodgerblue", linewidth=2)
    plt.plot(rec["center"][0], rec["center"][1], "s", color="dodgerblue", markersize=5)
plt.plot(
    [sx - size_src.x/2, sx + size_src.x/2],
    [sy - size_src.y/2, sy + size_src.y/2],
    color="red",
    linewidth=2,
)
plt.plot(sx, sy, "o", color="red", markersize=8, label="source horn")
plt.plot([], [], "s", color="dodgerblue", label="receiver horns")
plt.legend(loc="upper right")
plt.title("Phase 1.4: source and receiver horns")
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(image_dir, exist_ok=True)
plt.savefig(os.path.join(image_dir, "measure_1_4.png"), dpi=200, bbox_inches="tight")
#print("Wrote measure_1_4.png")

# ***Collect port signals***

sim.run(until_after_sources=mp.stop_when_dft_decayed(
    tol=1e-4, maximum_run_time=500))

freqs = np.array(mp.get_eigenmode_freqs(receivers[0]["monitor"]))
#print("Collected mode signals")
#print("a_out leaves the device through that horn. a_in enters it.")
#print(f"source horn is k={k_src}")
for rec in receivers:
    k_out = rec["k_out"]
    coeffs = sim.get_eigenmode_coefficients(
        rec["monitor"],
        [1],
        eig_parity=mp.ODD_Z,
        direction=mp.NO_DIRECTION,
        kpoint_func=lambda freq, band, k=k_out: k,
    )
    a_out = coeffs.alpha[0, :, 0]
    a_in = coeffs.alpha[0, :, 1]
    rec["a_out"] = a_out
    rec["a_in"] = a_in
    theta = port_centers[rec["k"]][2]
    role = "source + receiver" if rec["k"] == k_src else "receiver"
    #print(f"horn {rec['k']} ({role}), theta={theta*180/np.pi:.1f} deg")
    #print(f"{'f_GHz':>8} {'a_out_re':>12} {'a_out_im':>12} {'|a_out|^2':>12}"
    #      f" {'a_in_re':>12} {'a_in_im':>12} {'|a_in|^2':>12}")
    #for f, ao, ai in zip(freqs, a_out, a_in):
    #    print(f"{f*30:8.3f} {ao.real:12.4e} {ao.imag:12.4e} {abs(ao)**2:12.4e}"
    #          f" {ai.real:12.4e} {ai.imag:12.4e} {abs(ai)**2:12.4e}")

# ***S parameters***

# One column of S, with horn k_src driven. The reference is the inward wave
# on that horn. |S|^2 is the fraction of that incident power leaving each horn.
a_inc = receivers[k_src]["a_in"]
print(f"S parameters, horn {k_src} driven")
print(f"S_i{k_src} = a_out(i) / a_in({k_src})")
for rec in receivers:
    S = rec["a_out"] / a_inc
    theta = port_centers[rec["k"]][2]
    print(f"horn {rec['k']}, theta={theta*180/np.pi:.1f} deg, S_{rec['k']}{k_src}")
    print(f"{'f_GHz':>8} {'S_re':>12} {'S_im':>12} {'|S|':>12} {'|S|^2':>12}")
    for f, s in zip(freqs, S):
        print(f"{f*30:8.3f} {s.real:12.4e} {s.imag:12.4e} {abs(s):12.4e} {abs(s)**2:12.4e}")

