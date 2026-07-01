"""
Heterogeneity study: how permeability structure wrecks sweep efficiency
=======================================================================

Mobility ratio is one villain of a waterflood; heterogeneity is the other.
This study holds the fluids fixed (mu_o = 4 cp, M = 1.56) and compares three
permeability fields with the same average permeability (200 mD):

    1. Homogeneous: 200 mD everywhere (the base case).
    2. High-perm channel: a diagonal 1,000 mD channel connecting injector to
       producer through lower-perm background rock. Injected water
       short-circuits down the channel, breaks through early, and bypasses the
       off-channel oil.
    3. Correlated random field: log-normal permeability with a Dykstra-Parsons
       coefficient V = 0.7 and a finite correlation length, the standard
       statistical description of real reservoir heterogeneity. Water finds the
       connected high-perm pathways and fingers through them.

Keeping the mean permeability the same across cases makes the point cleanly:
single-phase deliverability is similar, but recovery is not, because what kills
a flood is the CONTRAST and CONNECTIVITY of permeability, not its average.

Run:  python heterogeneity_study.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from reservoir_sim import Config, Simulator


def make_channel(nx, ny, k_channel=1000.0, half_width=2, k_mean=200.0):
    """Diagonal high-perm channel from injector corner to producer corner.
    Background permeability is chosen so the field mean equals k_mean."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    in_channel = np.abs(i - j) <= half_width
    n_ch = in_channel.sum()
    k_bg = (k_mean * nx * ny - k_channel * n_ch) / (nx * ny - n_ch)
    k = np.full((nx, ny), k_bg)
    k[in_channel] = k_channel
    return k


def make_lognormal(nx, ny, V_dp=0.7, corr_cells=4.0, k_mean=200.0, seed=42):
    """Correlated log-normal permeability field.

    V_dp is the Dykstra-Parsons coefficient; for a log-normal distribution
    V_dp = 1 - exp(-sigma_lnk), so sigma_lnk = -ln(1 - V_dp). The field is
    built by Gaussian-smoothing white noise (correlation length ~ corr_cells
    cells), rescaling to the target sigma, and shifting to the target mean.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    z = gaussian_filter(rng.standard_normal((nx, ny)), sigma=corr_cells)
    z = (z - z.mean()) / z.std()
    sigma = -np.log(1.0 - V_dp)
    k = np.exp(sigma * z)
    return k * (k_mean / k.mean())


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    nx = ny = 31
    snap = 0.3
    cases = [
        ("Homogeneous", None),
        ("High-perm channel", make_channel(nx, ny)),
        ("Log-normal (V=0.7)", make_lognormal(nx, ny)),
    ]

    results = []
    print(f"{'case':<20}{'mean k (mD)':>12}{'BT PVI':>9}{'RF @ 1 PVI':>12}"
          f"{'MB error':>12}")
    for name, kf in cases:
        cfg = Config(nx=nx, ny=ny, k_field=kf, mu_o=4.0,
                     snap_PVI=(snap,), PVI_end=1.0)
        res = Simulator(cfg).run(verbose=False)
        results.append((name, cfg, res))
        print(f"{name:<20}{cfg.k_field.mean():>12.0f}"
              f"{res['breakthrough_PVI']:>9.3f}{res['RF']:>12.3f}"
              f"{res['mass_balance_error']:>12.1e}")

    # ---- figure 1: permeability fields and saturation at 0.3 PVI ------------
    extent = [0, 1000, 0, 1000]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    for col, (name, cfg, res) in enumerate(results):
        axk = axes[0, col]
        from matplotlib.colors import LogNorm
        imk = axk.imshow(cfg.k_field.T, origin='lower', extent=extent,
                         cmap='cividis', aspect='equal',
                         norm=LogNorm(vmin=30, vmax=1000))
        axk.set_title(f"{name}\nmean k = {cfg.k_field.mean():.0f} mD")
        axk.set_xlabel("x (ft)")
        if col == 0:
            axk.set_ylabel("y (ft)")
        fig.colorbar(imk, ax=axk, shrink=0.85, label="k (mD)")

        axs = axes[1, col]
        ims = axs.imshow(res['snaps'][snap].T, origin='lower', extent=extent,
                         vmin=cfg.Swc, vmax=1 - cfg.Sor, cmap='turbo',
                         aspect='equal')
        axs.set_title(f"$S_w$ at {snap:.1f} PVI  |  "
                      f"BT = {res['breakthrough_PVI']:.2f} PVI, "
                      f"RF@1PVI = {res['RF']:.2f}")
        axs.set_xlabel("x (ft)")
        if col == 0:
            axs.set_ylabel("y (ft)")
        fig.colorbar(ims, ax=axs, shrink=0.85, label="$S_w$")
    fig.suptitle("Same average permeability, very different floods: "
                 "contrast and connectivity control sweep",
                 fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig("figures/heterogeneity_fields.png", dpi=130, bbox_inches='tight')

    # ---- figure 2: recovery and water cut curves -----------------------------
    fig2, (axr, axw) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ['tab:gray', 'tab:red', 'tab:blue']
    for (name, cfg, res), col in zip(results, colors):
        h = res['hist']
        axr.plot(h['PVI'], h['RF'], color=col, lw=2, label=name)
        axw.plot(h['PVI'], h['wc'], color=col, lw=2, label=name)
    axr.set_xlabel("pore volumes injected"); axr.set_ylabel("recovery factor (movable)")
    axr.set_title("Oil recovery"); axr.legend(); axr.set_ylim(0, 1)
    axw.set_xlabel("pore volumes injected"); axw.set_ylabel("water cut $f_w$")
    axw.set_title("Water cut at producer"); axw.legend(); axw.set_ylim(0, 1)
    fig2.suptitle("Heterogeneity causes early breakthrough and bypassed oil "
                  "(fluids identical, M = 1.56)", fontsize=13, weight='bold')
    fig2.tight_layout()
    fig2.savefig("figures/heterogeneity_curves.png", dpi=130, bbox_inches='tight')
    print("Saved figures/heterogeneity_fields.png and "
          "figures/heterogeneity_curves.png")
