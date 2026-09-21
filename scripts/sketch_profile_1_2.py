# Plasma profile parameterization - build MEEP applicable plasma profile using cocentric stacked rings

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros
from scipy.optimize import brentq
from plasmeep.lib import Plasmeep as pm, WP
import meep as mp

#constants
R_p = 1.5 #in units of a
r = np.linspace(0,R_p,200)
a = 0.01

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

#n0 prior range choice
model = pm(a, res=20, dpml=5.0, nx=24, ny=24)

n0_low = 5e16 #lower density for 2 GHz in m^3 
n0_high = 1.5e18 #upper density for 11 GHz in m^3
wplow_rad_s = WP(n0_low) #angular plasma frequency (rad/s)
wphigh_rad_s = WP(n0_high)
fp_Hz_low = wplow_rad_s/(2*np.pi) #convert to Hz
fp_Hz_high = wphigh_rad_s/(2*np.pi)
wp_low_meep = model.Nondimensionalize_Freq(fp_Hz_low) #MEEP units conversion
wp_high_meep = model.Nondimensionalize_Freq(fp_Hz_high)

gamma_Hz = 1e9 #collision rate: gamma/2pi ~ 1 GHz
gamma_meep = model.Nondimensionalize_Freq(gamma_Hz)

print('n0 lower bound:', n0_low, ', n0 upper bound:', n0_high)
print('Lower wp (MEEP):', wp_low_meep, ', Upper wp (MEEP):', wp_high_meep, ', Gamma (MEEP):', gamma_meep)

#Beta sample range from uniform h
h = np.linspace(0.05, 0.8, 16)
beta = []
for hi in h:
    b = beta_from_h(hi)
    beta.append(b)
beta = np.array(beta)

n0 = 1.0  # normalize so center = 1; this is a shape plot
plt.figure()
for h, beta in zip(h, beta):
    plt.plot(
        r,
        bessel_profile(r, n0, beta, R_p),
        label=f"h={h:.2f}, β={beta:.2f}",
    )
plt.xlabel("r / a")
plt.ylabel("n_e / n0")
plt.legend(fontsize=8)
plt.title("Phase 1.2: shapes from h prior")
plt.savefig("profile_1_2_h_prior.png", dpi=200, bbox_inches="tight")
print("Wrote profile_1_2_h_prior.png")

#Cocentric drude shells for example profile 
N_shells = 12
n0_si = n0_low
h_ex = 0.4
beta_ex = beta_from_h(h_ex)

model_2 = pm(a, res=20, dpml=5.0, nx=24, ny=24)
gamma_Hz=1e9
gamma_meep=model_2.Nondimensionalize_Freq(gamma_Hz)

shell_r=[]
shell_n=[]

for i in range(N_shells, 0, -1):
    r_out=R_p*i/N_shells
    r_mid=R_p*(i-0.5)/N_shells

    n_i = float(bessel_profile(r_mid, n0_si, beta_ex, R_p))
    n_i = max(n_i, 1e10)

    fp_Hz = WP(n_i)/(2*np.pi)
    wp_meep = model_2.Nondimensionalize_Freq(fp_Hz)

    medium = model_2.Get_Med(1.0, wp=wp_meep, gamma=gamma_meep)
    model_2.geometry.append(
        mp.Cylinder(
            radius=r_out,
            material=medium,
            center=mp.Vector3(0,0,0),
        )
    )

    shell_r.append(r_mid)
    shell_n.append(n_i)
    print(f"shell {i:2d}: r_mid={r_mid:.3f}, n={n_i:.3e} m^-3, fp={fp_Hz/1e9:.2f} GHz")

r_plot = np.linspace(0,R_p,200)
plt.figure()
plt.plot(r_plot, bessel_profile(r_plot, n0_si, beta_ex, R_p), label="smooth Bessel")
plt.plot(shell_r, shell_n, 'o', label='shell samples')
plt.xlabel('r/a')
plt.ylabel('n_e(m^-3)')
plt.legend()
plt.title('Phase 1.2: shell discretization')
plt.savefig('profile_1_2_shells.png', dpi=200, bbox_inches='tight')
print('Wrote profile_1_2_shells.png')

Npix = 300
lim = R_p * 1.15
xs = np.linspace(-lim, lim, Npix)
ys = np.linspace(-lim, lim, Npix)
X, Y = np.meshgrid(xs, ys)
R = np.sqrt(X**2 + Y**2)
N_map = np.full(R.shape, np.nan)  # outside the plasma stays blank
for i in range(1, N_shells + 1):
    r_inner = R_p * (i - 1) / N_shells
    r_outer = R_p * i / N_shells
    r_mid = R_p * (i - 0.5) / N_shells
    n_i = max(float(bessel_profile(r_mid, n0_si, beta_ex, R_p)), 0.0)
    if i < N_shells:
        mask = (R >= r_inner) & (R < r_outer)
    else:
        mask = (R >= r_inner) & (R <= r_outer)
    N_map[mask] = n_i
plt.figure()
plt.imshow(
    N_map,
    extent=[-lim, lim, -lim, lim],
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="n_e (m^-3)")
plt.xlabel("x / a")
plt.ylabel("y / a")
plt.title("Phase 1.2: concentric shell densities")
plt.gca().set_aspect("equal")
plt.savefig("profile_1_2_rings.png", dpi=200, bbox_inches="tight")
print("Wrote profile_1_2_rings.png")
