import os
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa #numpy that the adjoint solver can differentiate
from plasmeep.lib import Plasmeep as pm


# ***Units and domain***
a = 0.01 # meters
res = 20 # pixels per a
dpml = 5.0 # PML thickness in units of a
nx = 52 # cell size in units of a, PML included
ny = 52

RUN_SIM = True #False: build and plot the geometry only. True: also run and get S-parameters.

# ***Scaffold and plasma***
R_p = 1.5 #plasma radius
r_in = 2.0 #vacuum hole around the plasma
eps_scaffold = 4.0 #placeholder uniform dielectric - design variable

# ***Horn dimensions (InversePMMDesign Add_INFOMW_Horn, TM orientation)***
# Lab horn sizes in meters, converted to units of a.
n_ports = 5
wall_t = 0.004/a #wall thickness
width_open = 0.104/a #outer width at the aperture
width_base = 0.048/a #outer width at the throat / feed waveguide
depth = 0.089/a #flare length, aperture to throat
w_feed_in = width_base - 2*wall_t #inner width of the feed waveguide

# ***Pentagon scaffolding***
# One horn aperture per side, so the side length is the aperture width.
side = width_open
apothem = side/(2*np.tan(np.pi/n_ports)) #center to middle of a side
R_pent = side/(2*np.sin(np.pi/n_ports)) #center to a vertex

model = pm(a, res, dpml, nx, ny) #creates plasmeep object holding the settings and list of simulation objects

# Side k faces theta_k = k*72 deg. Vertices sit halfway between.
#Build scaffolding in pentagon shape
port_thetas = [k*2*np.pi/n_ports for k in range(n_ports)]
pent_vertices = [
    mp.Vector3(R_pent*np.cos(t + np.pi/n_ports), R_pent*np.sin(t + np.pi/n_ports), 0)
    for t in port_thetas
]
#create dielectric scaffolding
model.geometry.append(
    mp.Prism(
        vertices=pent_vertices,
        height=mp.inf,
        axis=mp.Vector3(0, 0, 1),
        material=mp.Medium(epsilon=eps_scaffold),
    )
)
#vacuum hole
model.geometry.append(
    mp.Cylinder(
        radius=r_in,
        material=mp.Medium(epsilon=1.0), #vacuum hole
        center=mp.Vector3(0, 0, 0),
    )
)

# ***Horns***
#function creates horn with correct shape
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

    #points horn toward the scaffolding
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

# Each feed guide runs straight out until it is dpml/2 deep in the PML.
# A ray at angle theta hits the square |x|,|y| = h at h/max(|cos|,|sin|).
h_end = nx/2 - dpml/2
r_throat = apothem + depth #flare-to-feed junction, measured from the center

#placement loop for the horns
for k, theta in enumerate(port_thetas):
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
    print(f"horn {k}: theta = {np.degrees(theta):5.1f} deg, "
          f"feed length = {r_end - r_throat:.2f} a")

# ***Source horn***
# Per the student guide (Phase 1.1): TM polarization (Ez out of plane), one
# EigenModeSource per run at a source horn, broadband pulse over the probe
# band. The lab VNA can source from 2 non-adjacent horns (144 deg apart), so
# k_src = 0 now and k_src = 2 for the second drive later.
k_src = 0

#probe band in MEEP frequency units (f = 1 <-> 30 GHz at a = 1 cm)
#The metal feed guide cuts off at f = 1/(2*w_feed_in) = 0.125 (3.75 GHz),
#so the guide's 0.1 lower edge is raised to 0.14. Above 0.25 the feed also
#carries a second mode; eig_band=1 launches only the first.
f_min = 0.14
f_max = 0.3
fcen = 0.5*(f_min + f_max)
df = f_max - f_min

# Positions in the straight feed, measured back from the throat (flare-to-feed
# junction). In the source horn the monitor sits between the source and the
# flare, so it sees the incident wave (inward) and the reflection (outward)
# separately. Horn 0 has the shortest feed outside the PML (~4.9 a), so keep
# d_src below about 3.
d_src = 3.0 #source, a behind the throat
d_mon = 1.0 #port monitors, a behind the throat
n_freq = 41
freqs = np.linspace(f_min, f_max, n_freq) #frequencies that monitor records

#the source line
def feed_line(theta, r, margin=2.0/res):
    """Axis-aligned line across a horn's feed guide at radius r.

    Meep source lines lie on the grid axes. A horn that runs mostly along x
    is cut along y, and vice versa. The line stops 'margin' short of each
    metal wall, so MPB never sees the PEC.
    Returns (center, size, k_out); k_out points away from the device.
    """
    c, s = float(np.cos(theta)), float(np.sin(theta))
    span = (w_feed_in - 2*margin)/max(abs(c), abs(s))
    size = mp.Vector3(0, span, 0) if abs(c) >= abs(s) else mp.Vector3(span, 0, 0)
    return mp.Vector3(r*c, r*s, 0), size, mp.Vector3(c, s, 0)

# For a tilted horn the source line is axis-aligned and eig_kpoint points
# down the horn axis (direction=NO_DIRECTION). On the tilted horns the
# launched mode is slightly impure; that is fine, because the incident wave
# is measured at the source-horn port and everything is normalized to it.
theta_src = port_thetas[k_src]
src_center, src_size, k_out_src = feed_line(theta_src, r_throat + d_src)

#soure along source line
model.sources = [mp.EigenModeSource(
    src=mp.GaussianSource(frequency=fcen, fwidth=df),
    center=src_center,
    size=src_size,
    direction=mp.NO_DIRECTION,
    eig_kpoint=k_out_src.scale(-1), #launch toward the plasma
    eig_band=1,
    eig_parity=mp.ODD_Z, #Ez polarization (TM)
    eig_match_freq=True,
)]

# ***Port monitors (raw DFT fields on tilted lines)***
# Each port records the raw Fourier-transformed Ez, Hx, Hy in a small
# axis-aligned box around its feed guide (mpa.FourierFields, the adjoint
# solver's differentiable DFT monitor). Afterwards the fields are
# interpolated onto a line perpendicular to the horn axis, at the horn's
# true angle. From those line fields we get:
#   port_flux  - raw Poynting power through the line (the advisor's method)
#   port_modes - complex outgoing/incoming amplitudes of the fundamental
#                parallel-plate mode cos(pi*u/w), found by projection
# Both are plain autograd math on the DFT fields, so the same functions can
# be used inside an adjoint objective.

fc_feed = 1/(2*w_feed_in) #cutoff of the fundamental feed-guide mode

def bilinear_matrix(xs, ys, pts):
    """Matrix W so that (W @ F.ravel()) interpolates grid values F[ix, iy]
    at the points pts (N x 2), bilinearly."""
    nxs, nys = len(xs), len(ys)
    W = np.zeros((len(pts), nxs*nys))
    for p, (px, py) in enumerate(pts):
        i = int(np.clip(np.searchsorted(xs, px) - 1, 0, nxs - 2))
        j = int(np.clip(np.searchsorted(ys, py) - 1, 0, nys - 2))
        tx = (px - xs[i])/(xs[i+1] - xs[i])
        ty = (py - ys[j])/(ys[j+1] - ys[j])
        for di, dj, wt in ((0, 0, (1-tx)*(1-ty)), (1, 0, tx*(1-ty)),
                           (0, 1, (1-tx)*ty), (1, 1, tx*ty)):
            W[p, (i+di)*nys + (j+dj)] = wt
    return W

class TiltedPort:
    """Raw-field monitor on a line across a feed guide, at the horn's angle.

    n: outward unit normal (along the horn axis, away from the device)
    t: unit vector along the line, across the guide (t = z x n)
    u: positions along the line, -w/2..w/2, with trapezoid weights du
    """
    def __init__(self, sim, theta, r, pad=3.0/res):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        self.sim = sim
        self.theta = theta
        self.n = np.array([c, s])
        self.t = np.array([-s, c])
        self.center = r*self.n
        n_pts = int(np.ceil(2*res*w_feed_in)) + 1 #half-pixel spacing
        self.u = np.linspace(-w_feed_in/2, w_feed_in/2, n_pts)
        self.du = np.full(n_pts, self.u[1] - self.u[0])
        self.du[[0, -1]] *= 0.5
        self.pts = self.center[None, :] + self.u[:, None]*self.t[None, :]
        lo = self.pts.min(axis=0) - pad
        hi = self.pts.max(axis=0) + pad
        self.box = mp.Volume(center=mp.Vector3(*(0.5*(lo + hi))),
                             size=mp.Vector3(*(hi - lo)))
        self.fields = [mpa.FourierFields(sim, self.box, comp) #voxel-centered
                       for comp in (mp.Ez, mp.Hx, mp.Hy)]
        self.W = None

    def register_monitors(self, frequencies):
        #Not needed inside mpa.OptimizationProblem, which registers them itself.
        for ff in self.fields:
            ff.register_monitors(frequencies)

    def _build_interp(self):
        #Grid coordinates of the DFT arrays; available once the monitor exists.
        xs, ys, _, _ = self.sim.get_array_metadata(dft_cell=self.fields[0]._monitor)
        self.grid_shape = (len(xs), len(ys))
        self.W = bilinear_matrix(np.asarray(xs), np.asarray(ys), self.pts)

    def line_fields(self, Ez, Hx, Hy):
        """Ez and H_t (H along t) on the tilted line, shape (n_freq, n_pts)."""
        if self.W is None:
            self._build_interp()
        nf = Ez.shape[0]
        interp = lambda F: npa.dot(npa.reshape(F, (nf, -1)), self.W.T)
        Ez_l = interp(Ez)
        Ht_l = interp(Hx)*self.t[0] + interp(Hy)*self.t[1]
        return Ez_l, Ht_l

def port_flux(port, Ez_l, Ht_l):
    """Net Poynting power out through the line (away from the device).
    S.n = 0.5 Re(E x H*).n = -0.5 Re(Ez conj(H_t)) for Ez polarization."""
    return -0.5*npa.real(npa.sum(Ez_l*npa.conj(Ht_l)*port.du, axis=1))

def port_modes(port, Ez_l, Ht_l, freqs):
    """Complex amplitudes (c_out, c_in) of the fundamental feed-guide mode.

    Mode profile m(u) = cos(pi*u/w). For a wave leaving the device,
    H_t = -(beta/omega) Ez; for one arriving, H_t = +(beta/omega) Ez.
    beta/omega is exact at every frequency (the guide is analytic).
    Ignores the second, antisymmetric mode (above 0.25), which has zero
    overlap with m(u)."""
    m = np.cos(np.pi*port.u/w_feed_in)
    norm = np.sum(m*m*port.du)
    A = npa.sum(Ez_l*m*port.du, axis=1)/norm
    B = npa.sum(Ht_l*m*port.du, axis=1)/norm
    omega_over_beta = 1/np.sqrt(1 - (fc_feed/np.asarray(freqs))**2)
    c_out = 0.5*(A - omega_over_beta*B)
    c_in = 0.5*(A + omega_over_beta*B)
    return c_out, c_in

def mode_power(port, c, freqs):
    """Power carried by mode amplitude c: 0.5 (beta/omega) |c|^2 int m^2 du."""
    m = np.cos(np.pi*port.u/w_feed_in)
    beta_over_omega = np.sqrt(1 - (fc_feed/np.asarray(freqs))**2)
    return 0.5*beta_over_omega*npa.abs(c)**2*np.sum(m*m*port.du)

def s_parameters(*dft_arrays):
    """S-parameters from the raw DFT arrays of all ports.

    dft_arrays = (Ez_0, Hx_0, Hy_0, Ez_1, Hx_1, Hy_1, ...), in port order,
    the same order as objective_arguments = [ff for p in ports for ff in p.fields].
    Returns
      S: complex mode S-parameters, S[k] = c_out,k / c_in,src
      T: raw-Poynting power ratios, T[k] = P_k / P_inc  (T[k_src] = reflected/incident)
    """
    lines = [ports[k].line_fields(*dft_arrays[3*k:3*k + 3]) for k in range(n_ports)]
    modes = [port_modes(ports[k], *lines[k], freqs) for k in range(n_ports)]
    c_inc = modes[k_src][1]
    S = npa.stack([modes[k][0]/c_inc for k in range(n_ports)])

    #In the source horn the net flux is P_refl - P_inc, so add P_inc back.
    P_inc = mode_power(ports[k_src], c_inc, freqs)
    P = [port_flux(ports[k], *lines[k]) for k in range(n_ports)]
    P[k_src] = P[k_src] + P_inc
    T = npa.stack([P[k]/P_inc for k in range(n_ports)])
    return S, T

sim = model.Get_Sim()

ports = [TiltedPort(sim, theta, r_throat + d_mon) for theta in port_thetas]
for p in ports:
    p.register_monitors(freqs)

# ***Plot***

# Meep's plot2D scales its gray map from the lowest to the highest epsilon.
# The metal walls count as a huge negative epsilon, so vacuum (1) and the
# scaffold (4) both land at the dark end and look the same. Instead, grab
# the epsilon grid and color the metal separately.
sim.init_sim()
eps = np.real(sim.get_array(center=mp.Vector3(), size=sim.cell_size,
                            component=mp.Dielectric))
metal = ~np.isfinite(eps) | (eps < 0.5) | (eps > 1e3)
eps_plot = np.ma.masked_where(metal, eps)

cmap = plt.cm.Blues.copy()
cmap.set_bad("dimgray") #metal horn walls

fig, ax = plt.subplots(figsize=(7, 7))
extent = [-nx/2, nx/2, -ny/2, ny/2]
im = ax.imshow(eps_plot.T, origin="lower", extent=extent, cmap=cmap,
               vmin=1, vmax=eps_scaffold, interpolation="nearest")
fig.colorbar(im, ax=ax, shrink=0.8, label="relative permittivity")

# Inner edge of the PML
ax.add_patch(plt.Rectangle((-nx/2 + dpml, -ny/2 + dpml), nx - 2*dpml, ny - 2*dpml,
                           fill=False, color="green", linestyle=":", linewidth=1,
                           label="PML edge"))
# The plasma is not in the geometry yet. Outline where it will go.
ax.add_patch(plt.Circle((0, 0), R_p, fill=False, color="orange",
                        linestyle="--", linewidth=1.5, label="plasma (R_p)"))

# Source line in red
ax.plot([src_center.x - src_size.x/2, src_center.x + src_size.x/2],
        [src_center.y - src_size.y/2, src_center.y + src_size.y/2],
        color="red", linewidth=2.5, label=f"source (horn {k_src})")
# Tilted port lines in magenta
for k, p in enumerate(ports):
    ax.plot(p.pts[:, 0], p.pts[:, 1], color="magenta", linewidth=2,
            label="port monitors" if k == 0 else None)
ax.plot([], [], "s", color="dimgray", label="metal horn walls")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax.set_xlabel("x (a)")
ax.set_ylabel("y (a)")
ax.set_title("Phase 1.5 geometry: pentagon scaffold, InfoMW horns")

image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(image_dir, exist_ok=True)
out = os.path.join(image_dir, "geometry_1_5.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Wrote", out)
plt.show()

# ***Run and S-parameters***
# Solid lines: |S|^2 from the mode projection. Dotted: raw Poynting ratio.
# Below 0.25 (7.5 GHz) only one mode propagates and the two should agree.
# Above it, the Poynting ratio also counts power in the second mode.

if RUN_SIM:
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6))

    dft_arrays = [ff() for p in ports for ff in p.fields]
    S, T = s_parameters(*dft_arrays)

    #Save the results for this drive
    np.savez(os.path.join(image_dir, f"S_src{k_src}.npz"),
             freqs=freqs, freqs_GHz=freqs*30, S=S, T=T,
             port_thetas=np.array(port_thetas), k_src=k_src)
    for k in range(n_ports):
        print(f"horn {k} at fcen: |S_{k}{k_src}|^2 = {np.abs(S[k, n_freq//2])**2:.4f}, "
              f"Poynting ratio = {T[k, n_freq//2]:.4f}")

    #One panel per horn, plus an energy-check panel. All panels share axes.
    f_GHz = freqs*30
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for k in range(n_ports):
        ax = axes[k]
        ax.plot(f_GHz, 10*np.log10(np.abs(S[k])**2), "-", color="C0",
                label="mode projection |S|²")
        ax.plot(f_GHz, 10*np.log10(np.abs(T[k])), ":", color="C1",
                label="raw Poynting")
        ax.axvline(7.5, color="gray", linewidth=0.8, linestyle="--")
        role = "reflection, source horn" if k == k_src else "transmission"
        ax.set_title(f"horn {k} ({np.degrees(port_thetas[k]):.0f}°): "
                     f"S_{k}{k_src}, {role}")
        ax.grid(alpha=0.3)

    #Last panel: energy check
    ax = axes[n_ports]
    ax.plot(f_GHz, 10*np.log10(np.sum(np.abs(S)**2, axis=0)), "k-")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(7.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("sum of |S|² over all horns (energy check)")
    ax.grid(alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("frequency (GHz)")
    for ax in axes[::2]:
        ax.set_ylabel("power fraction (dB)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Drive from horn {k_src} "
                 f"(dashed line: second feed mode turns on at 7.5 GHz)")
    fig.tight_layout()
    out = os.path.join(image_dir, f"S_src{k_src}_per_horn.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote", out)
    plt.show()