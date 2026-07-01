"""
Mobility-ratio study: effect of the end-point mobility ratio on sweep
=====================================================================

Runs the quarter five-spot at four oil viscosities, so four end-point mobility
ratios M, holding everything else fixed. M > 1 means the injected water is more
mobile than the oil it displaces: it fingers ahead, breaks through early, and
leaves oil behind. M < 1 sweeps closer to a piston.

Run:  python mobility_ratio_study.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from reservoir_sim import Config, Simulator

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    snap = 0.3
    mu_oils = [0.5, 2.0, 5.0, 10.0]

    results = []
    print(f"{'mu_o (cp)':>10}{'M':>8}{'BT PVI':>9}{'RF @ 1 PVI':>12}{'MB error':>12}")
    for mu in mu_oils:
        cfg = Config(mu_o=mu, snap_PVI=(snap,), PVI_end=1.0)
        res = Simulator(cfg).run(verbose=False)
        results.append((cfg, res))
        print(f"{mu:>10.1f}{cfg.mobility_ratio:>8.2f}"
              f"{res['breakthrough_PVI']:>9.3f}{res['RF']:>12.3f}"
              f"{res['mass_balance_error']:>12.1e}")

    # ---- figure 1: saturation at 0.3 PVI, all four cases ---------------------
    extent = [0, 1000, 0, 1000]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (cfg, res) in zip(axes, results):
        im = ax.imshow(res['snaps'][snap].T, origin='lower', extent=extent,
                       vmin=cfg.Swc, vmax=1 - cfg.Sor, cmap='turbo',
                       aspect='equal')
        fav = "favorable" if cfg.mobility_ratio < 1 else "unfavorable"
        ax.set_title(f"M = {cfg.mobility_ratio:.2f} ({fav})\n"
                     f"$\\mu_o$ = {cfg.mu_o:.1f} cp", fontsize=11)
        ax.set_xlabel("x (ft)")
    axes[0].set_ylabel("y (ft)")
    fig.suptitle("Sweep at 0.3 PVI: rising mobility ratio worsens the flood",
                 fontsize=14, weight='bold')
    fig.colorbar(im, ax=list(axes), shrink=0.75, label="$S_w$")
    fig.savefig("figures/mobility_saturation.png", dpi=130, bbox_inches='tight')

    # ---- figure 2: recovery and water cut vs PVI -----------------------------
    fig2, (axr, axw) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for (cfg, res), col in zip(results, colors):
        h = res['hist']
        lbl = f"M = {cfg.mobility_ratio:.2f}"
        axr.plot(h['PVI'], h['RF'], color=col, lw=2, label=lbl)
        axw.plot(h['PVI'], h['wc'], color=col, lw=2, label=lbl)
    axr.set_xlabel("pore volumes injected"); axr.set_ylabel("recovery factor (movable)")
    axr.set_title("Oil recovery"); axr.legend(); axr.set_ylim(0, 1)
    axw.set_xlabel("pore volumes injected"); axw.set_ylabel("water cut $f_w$")
    axw.set_title("Water cut at producer"); axw.legend(); axw.set_ylim(0, 1)
    fig2.suptitle("Higher mobility ratio: earlier breakthrough, lower recovery",
                  fontsize=13, weight='bold')
    fig2.savefig("figures/mobility_curves.png", dpi=130, bbox_inches='tight')
    print("Saved figures/mobility_saturation.png and figures/mobility_curves.png")
