"""
Validation: numerical simulator vs the Buckley-Leverett analytical solution
===========================================================================

The Buckley-Leverett problem (1D, incompressible, immiscible displacement with
zero capillary pressure) has an exact analytical solution, so it is the standard
benchmark for a two-phase transport scheme. This script runs the simulator on a
1D grid (ny = 1) with the same fluid properties as the base 2D model and
compares three things against the analytical answer:

    1. The saturation profile Sw(x) at fixed times before breakthrough
       (rarefaction fan + shock front, from the Welge tangent construction).
    2. The breakthrough time in pore volumes injected: PVI_bt = 1 / f_w'(Swf).
    3. The recovery factor at breakthrough (Welge average saturation).

Agreement here demonstrates that the upstream-weighted explicit transport and
the implicit pressure solve reproduce the correct front speed and shock
saturation. The numerical front is slightly smeared over a few cells because
first-order upwinding is diffusive; that smearing shrinks as the grid is
refined, which is also shown.

Run:  python validate_buckley_leverett.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from reservoir_sim import Config, Simulator, BBL_TO_FT3


# ----------------------------------------------------------------------------
# Analytical Buckley-Leverett solution (Welge tangent construction)
# ----------------------------------------------------------------------------
def bl_analytical(sim, n=4001):
    """Return the pieces of the analytical solution for the sim's fluid props.

    xD(S, tD) = tD * f_w'(S) for S >= Swf (the rarefaction),
    with a shock from Swf down to Swc at xD_front = tD * f_w'(Swf).
    tD is dimensionless time in total pore volumes injected (PVI).
    """
    c = sim.cfg
    S = np.linspace(c.Swc, 1.0 - c.Sor, n)
    fw = sim.frac_flow(S)
    dfw = np.gradient(fw, S)

    # Welge tangent from (Swc, fw=0): the shock saturation Swf maximizes the
    # secant slope (fw(S) - 0) / (S - Swc).
    with np.errstate(divide='ignore', invalid='ignore'):
        secant = np.where(S > c.Swc, fw / (S - c.Swc), 0.0)
    i_f = int(np.argmax(secant))
    Swf = S[i_f]                       # front (shock) saturation
    v_shock = secant[i_f]              # dimensionless shock speed = fw'(Swf)

    PVI_bt = 1.0 / v_shock             # breakthrough, pore volumes injected

    # Welge average saturation behind the front at breakthrough:
    # Swbar = Swf + (1 - fw(Swf)) / fw'(Swf)
    Swbar_bt = Swf + (1.0 - fw[i_f]) / v_shock
    RF_bt = (Swbar_bt - c.Swc) / (1.0 - c.Swc - c.Sor)   # movable-oil RF

    return {'S': S, 'fw': fw, 'dfw': dfw, 'Swf': Swf,
            'v_shock': v_shock, 'PVI_bt': PVI_bt, 'RF_bt': RF_bt}


def bl_profile(sim, ana, PVI, nxD=2000):
    """Analytical Sw(xD) at dimensionless time tD = PVI."""
    c = sim.cfg
    S, dfw, Swf = ana['S'], ana['dfw'], ana['Swf']
    # rarefaction: S from Swmax down to Swf maps to xD = PVI * fw'(S).
    mask = S >= Swf
    S_r = S[mask][::-1]                # decreasing S with increasing x
    xD_r = PVI * dfw[mask][::-1]
    xD_front = PVI * ana['v_shock']

    xD = np.linspace(0.0, 1.0, nxD)
    Sw = np.full(nxD, c.Swc)
    behind = xD <= xD_front
    Sw[behind] = np.interp(xD[behind], xD_r, S_r)
    return xD, Sw


# ----------------------------------------------------------------------------
# Numerical 1D runs
# ----------------------------------------------------------------------------
def run_1d(nx, snap_PVI, PVI_end):
    cfg = Config(nx=nx, ny=1, Lx=1000.0, Ly=1000.0, h=20.0,
                 inj_cell=(0, 0), prod_cell=(nx - 1, 0),
                 snap_PVI=tuple(snap_PVI), PVI_end=PVI_end, cfl=0.5)
    sim = Simulator(cfg)
    res = sim.run(verbose=False)
    return sim, res


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    snap_PVI = [0.15, 0.30]
    nx_main = 200

    sim, res = run_1d(nx_main, snap_PVI, PVI_end=0.9)
    ana = bl_analytical(sim)

    # ---- figure: profiles at two times, plus a grid refinement panel -------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)

    colors = {snap_PVI[0]: 'tab:blue', snap_PVI[1]: 'tab:red'}
    ax = axes[0]
    for tgt in snap_PVI:
        xD_num = (np.arange(nx_main) + 0.5) / nx_main
        ax.plot(xD_num, res['snaps'][tgt][:, 0], 'o', ms=3,
                color=colors[tgt], alpha=0.6,
                label=f"numerical, {tgt:.2f} PVI")
        xD_a, Sw_a = bl_profile(sim, ana, tgt)
        ax.plot(xD_a, Sw_a, '-', color=colors[tgt], lw=2,
                label=f"analytical, {tgt:.2f} PVI")
    ax.axhline(ana['Swf'], color='k', ls=':', lw=1)
    ax.text(0.985, ana['Swf'] + 0.012, f"$S_{{wf}}$ = {ana['Swf']:.3f}",
            ha='right', fontsize=9)
    ax.set_xlabel("dimensionless distance $x_D$")
    ax.set_ylabel("water saturation $S_w$")
    ax.set_title(f"Saturation profiles ({nx_main} cells)")
    ax.legend(fontsize=9)

    # ---- grid refinement: front sharpens toward the analytical shock -------
    ax = axes[1]
    tgt = snap_PVI[1]
    for nx_i, col in zip([50, 100, 200, 400], plt.cm.plasma([0.15, 0.4, 0.65, 0.85])):
        _, r_i = run_1d(nx_i, [tgt], PVI_end=tgt + 0.05)
        xD_num = (np.arange(nx_i) + 0.5) / nx_i
        ax.plot(xD_num, r_i['snaps'][tgt][:, 0], '-', color=col, lw=1.8,
                label=f"{nx_i} cells")
    xD_a, Sw_a = bl_profile(sim, ana, tgt)
    ax.plot(xD_a, Sw_a, 'k--', lw=2, label="analytical")
    ax.set_xlim(0.3, 0.85)
    ax.set_xlabel("dimensionless distance $x_D$")
    ax.set_title(f"Grid refinement at {tgt:.2f} PVI\n"
                 "(first-order upwind smearing shrinks with $\\Delta x$)")
    ax.legend(fontsize=9)

    # ---- breakthrough and recovery scalars ----------------------------------
    hist = res['hist']
    PVI_bt_num = res['breakthrough_PVI']
    i_bt = np.argmax(hist['wc'] > 0.01)
    RF_bt_num = hist['RF'][i_bt]

    ax = axes[2]
    ax.plot(hist['PVI'], hist['wc'], 'b-', lw=2, label="numerical water cut")
    ax.axvline(ana['PVI_bt'], color='k', ls='--', lw=1.5,
               label=f"analytical breakthrough = {ana['PVI_bt']:.3f} PVI")
    ax.set_xlabel("pore volumes injected")
    ax.set_ylabel("water cut $f_w$")
    ax.set_title("Breakthrough timing")
    ax.legend(fontsize=9)

    fig.suptitle("Validation against the Buckley-Leverett analytical solution",
                 fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig("figures/validation_buckley_leverett.png", dpi=130,
                bbox_inches='tight')

    # ---- console report ------------------------------------------------------
    err_bt = (PVI_bt_num - ana['PVI_bt']) / ana['PVI_bt']
    err_rf = (RF_bt_num - ana['RF_bt']) / ana['RF_bt']
    print("Buckley-Leverett validation (1D, "
          f"{nx_main} cells, M = {sim.cfg.mobility_ratio:.2f})")
    print(f"  shock saturation Swf (analytical)        : {ana['Swf']:.4f}")
    print(f"  breakthrough PVI   analytical / numerical: "
          f"{ana['PVI_bt']:.4f} / {PVI_bt_num:.4f}  ({err_bt:+.1%})")
    print(f"  RF at breakthrough analytical / numerical: "
          f"{ana['RF_bt']:.4f} / {RF_bt_num:.4f}  ({err_rf:+.1%})")
    print(f"  water mass balance error                 : "
          f"{res['mass_balance_error']:+.2e}")
    print("Saved figures/validation_buckley_leverett.png")
