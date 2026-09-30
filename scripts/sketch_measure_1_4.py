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
r_in = 2.0 #scaffold inner radius (vacuum hole around the plasma)
eps_scaffold = 4.0 #placeholder uniform dielectricc - design variable

#  ***Horn dimensions (InversePMMDesign Add_INFOMW_Horn, TM orientation)***
# Metal-walled horn that matches the lab's microwave horns. Sizes are in
# meters in the original and are converted to units of a here.
n_ports = 5
wall_t = 0.004/a #wall thickness
width_open = 0.104/a #outer width at the aperture
width_base = 0.048/a #outer width at the throat / feed waveguide
depth = 0.089/a #flare length, aperture to throat
w_feed_in = width_base - 2*wall_t #inner width of the feed waveguide

#  ***Pentagon scaffold***
# One horn aperture per side, so the side length is the aperture width.
side = width_open
apothem = side/(2*np.tan(np.pi/n_ports)) #center to middle of a side
R_pent = side/(2*np.sin(np.pi/n_ports)) #center to a vertex

#domain size of a. Feed guides run straight out into the PML, as in the
#InversePMMDesign scripts, so the cell is sized around the horns.
nx = 52
ny = 52

#probe band in MEEP frequency units
#The metal feed guide cuts off below f = 1/(2*w_feed_in) = 0.125 (3.75 GHz),
#so f_min sits just above cutoff. Above 0.25 the feed also carries mode 2.
f_min = 0.14
f_max = 0.3

#print('a =', a, 'm')
#print('f=1 in MEEP <->', 3e8/a/1e9, 'GHz')
#print('probe band:', f_min*30, 'to', f_max*30, 'GHz')

#create empty simulation 'box'
model = pm(a, res, dpml, nx, ny)

# Side k faces the direction theta_k = k*72 deg. Vertices sit halfway between.
port_thetas = [k*2*np.pi/n_ports for k in range(n_ports)]
pent_vertices = np.array([
    [R_pent*np.cos(t + np.pi/n_ports), R_pent*np.sin(t + np.pi/n_ports), 0]
    for t in port_thetas
])
model.Add_Prism(pent_vertices, eps=eps_scaffold)
model.geometry.append(
    mp.Cylinder(
        radius=r_in,
        material=model.Get_Med(1.0),  # vacuum hole
        center=mp.Vector3(0, 0, 0),
    )
)

#  ***Horn build***

def add_infomw_horn(model, open_cen, horn_dir, entrance_length,
                    wall_t=0.4, width_open=10.4, width_base=4.8, depth=8.9):
    """Metal horn walls, ported from InversePMMDesign PMMI.Add_INFOMW_Horn.

    All lengths are in units of a (defaults are the lab horn at a = 1 cm).
    open_cen: (x, y) center of the aperture
    horn_dir: unit vector pointing out of the aperture, toward the device
    entrance_length: length of the straight feed waveguide behind the flare
    Appends four PEC prisms to model.geometry: two flared walls, two feed walls.
    """
    cx, cy = float(open_cen[0]), float(open_cen[1])
    dx, dy = float(horn_dir[0]), float(horn_dir[1])
    ox, oy = dy, -dx #unit vector across the horn
    back = depth + entrance_length

    def pt(across, along):
        #point at 'across' from the horn axis and 'along' behind the aperture
        return mp.Vector3(cx + across*ox - along*dx, cy + across*oy - along*dy, 0)

    for sgn in (1, -1):
        flare = [
            pt(sgn*width_open/2, 0),
            pt(sgn*(width_open/2 - wall_t), 0),
            pt(sgn*(width_base/2 - wall_t), depth),
            pt(sgn*width_base/2, depth),
        ]
        feed = [
            pt(sgn*(width_base/2 - wall_t), depth),
            pt(sgn*width_base/2, depth),
            pt(sgn*width_base/2, back),
            pt(sgn*(width_base/2 - wall_t), back),
        ]
        for verts in (flare, feed):
            model.geometry.append(
                mp.Prism(
                    vertices=verts,
                    height=mp.inf,
                    axis=mp.Vector3(0, 0, 1),
                    material=mp.perfect_electric_conductor,
                )
            )

# Run each feed guide straight out until it is dpml/2 deep in the PML.
# A ray at angle theta hits the square |x|,|y| = h at h/max(|cos|,|sin|).
h_end = nx/2 - dpml/2
r_throat = apothem + depth #radial position of the flare-to-feed junction

port_centers = [] #receiver point on each feed guide, and its angle
for theta in port_thetas:
    c, s = np.cos(theta), np.sin(theta)
    r_end = h_end/max(abs(c), abs(s))
    add_infomw_horn(
        model,
        open_cen=(apothem*c, apothem*s),
        horn_dir=(-c, -s),
        entrance_length=r_end - r_throat,
        wall_t=wall_t, width_open=width_open,
        width_base=width_base, depth=depth,
    )
    r_rec = r_throat + 1.0 #1 a into the feed, where the mode is clean
    port_centers.append((r_rec*c, r_rec*s, theta))

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

def horn_cut(theta):
    """Axis-aligned line across a horn, and the outward radial k-point.

    Meep monitor lines lie on the grid axes. A horn that runs mostly along x
    is cut along y; a horn that runs mostly along y is cut along x. The mode
    still propagates along the horn, set by this k-point with NO_DIRECTION.
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    # Oblique chord across the feed guide, ending halfway into each wall.
    span = (w_feed_in + wall_t)/max(abs(c), abs(s))
    if abs(c) >= abs(s):
        size = mp.Vector3(0, span, 0)
    else:
        size = mp.Vector3(span, 0, 0)
    k_out = mp.Vector3(c, s, 0)
    return size, k_out

cx0, cy0, theta0 = port_centers[k_src]
r_src = r_throat + 2.5 #outboard of the receiver line, still in the feed
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
    print(f"horn {k}: theta={theta*180/np.pi:.1f} deg, receiver")

print(f"source horn k={k_src} at {theta0*180/np.pi:.1f} deg, launching inward")

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
plt.title("Phase 1.4: pentagon scaffold, InfoMW horns")
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(image_dir, exist_ok=True)
plt.savefig(os.path.join(image_dir, "measure_1_4.png"), dpi=200, bbox_inches="tight")
print("Wrote measure_1_4.png")

# ***Collect port signals***

sim.run(until_after_sources=mp.stop_when_dft_decayed(
    tol=1e-4, maximum_run_time=500))

freqs = np.array(mp.get_eigenmode_freqs(receivers[0]["monitor"]))
print("Collected mode signals")
print("a_out leaves the device through that horn. a_in enters it.")
print(f"source horn is k={k_src}")
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
    print(f"horn {rec['k']} ({role}), theta={theta*180/np.pi:.1f} deg")
    print(f"{'f_GHz':>8} {'a_out_re':>12} {'a_out_im':>12} {'|a_out|^2':>12}"
          f" {'a_in_re':>12} {'a_in_im':>12} {'|a_in|^2':>12}")
    for f, ao, ai in zip(freqs, a_out, a_in):
        print(f"{f*30:8.3f} {ao.real:12.4e} {ao.imag:12.4e} {abs(ao)**2:12.4e}"
              f" {ai.real:12.4e} {ai.imag:12.4e} {abs(ai)**2:12.4e}")

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