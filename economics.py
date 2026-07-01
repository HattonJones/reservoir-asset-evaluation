"""
Layer 2 - Asset economics: from saturation fronts to NPV
========================================================

Takes the production stream from the physics simulator and runs it through a
discounted cash flow model, then optimizes the injection rate on NPV instead
of recovery. The point of the exercise is the A&D lesson: the design that
recovers the most oil is not the design that creates the most value.

How the physics feeds the economics
-----------------------------------
For incompressible flow the saturation solution depends only on pore volumes
injected, not on the clock. Doubling the injection rate produces the exact same
water cut and recovery curves versus PVI, just compressed in time:

    t(PVI) = PVI * PV_total / (5.615 * Q_inj)

So the simulator is run ONCE, and each candidate injection rate reuses that
solution with its own time axis. This is a deliberate, physics-justified
shortcut (it would not hold with compressibility or rate-dependent physics),
and it makes the rate optimization essentially free.

The trade-off that creates an interior NPV optimum
--------------------------------------------------
    - Higher rate pulls the same barrels forward in time -> less discounting.
    - Higher rate requires bigger injection/handling facilities -> more capex.
    - Fixed monthly opex burns longer at low rates -> earlier economic limit.

Production stops at the ECONOMIC LIMIT: the first month operating cash flow
goes negative. Reserves are what is recoverable ECONOMICALLY, so EUR is an
output of the economics, not just the physics.

Run:  python economics.py
"""

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from reservoir_sim import Config, Simulator, BBL_TO_FT3

DAYS_PER_MONTH = 30.4375


# ----------------------------------------------------------------------------
# 1. ECONOMIC ASSUMPTIONS
# ----------------------------------------------------------------------------
@dataclass
class EconInputs:
    oil_price: float = 70.0          # $/bbl, flat deck
    nri: float = 0.75                # net revenue interest (100% WI, 25% royalty)
    disc_rate: float = 0.10          # annual discount rate (NPV10)
    capex_dc: float = 6.0e6          # drill & complete, injector + producer, $
    capex_fac_base: float = 0.5e6    # fixed facilities, $
    capex_fac_per_bpd: float = 1500. # injection/handling capacity, $ per bbl/d
    opex_fixed: float = 30000.0      # fixed field opex, $/month
    opex_oil: float = 5.0            # variable lifting cost, $/bbl oil
    water_cost: float = 2.0          # $/bbl water handled (injected + produced)
    max_life_years: float = 30.0

    def capex(self, Q_inj):
        return self.capex_dc + self.capex_fac_base + self.capex_fac_per_bpd * Q_inj


# ----------------------------------------------------------------------------
# 2. CASH FLOW ENGINE
# ----------------------------------------------------------------------------
def monthly_volumes(res, Q_inj):
    """Rescale the (rate-independent) PVI solution to this injection rate and
    aggregate to monthly oil, produced water, and injected water volumes."""
    hist = res['hist']
    PV_total = res['PV_total']
    t_days = hist['PVI'] * PV_total / (BBL_TO_FT3 * Q_inj)

    # cumulative produced volumes vs PVI are rate independent, so build them
    # from the reference run and only the time axis depends on Q_inj
    dPVI = np.diff(hist['PVI'], prepend=0.0)
    scale = PV_total / (BBL_TO_FT3 * res['cfg'].Q_inj)
    cum_oil = np.cumsum((1.0 - hist['wc']) * hist['qt_prod'] * dPVI) * scale
    cum_wat = np.cumsum(hist['wc'] * hist['qt_prod'] * dPVI) * scale

    n_months = int(np.ceil(t_days[-1] / DAYS_PER_MONTH))
    edges = np.arange(n_months + 1) * DAYS_PER_MONTH
    oil_m = np.diff(np.interp(edges, t_days, cum_oil, left=0.0))
    watp_m = np.diff(np.interp(edges, t_days, cum_wat, left=0.0))
    inj_m = np.diff(np.minimum(edges, t_days[-1])) * Q_inj
    return oil_m, watp_m, inj_m


def evaluate(res, Q_inj, econ: EconInputs):
    """Full DCF for one injection rate. Returns metrics and monthly series."""
    oil_m, watp_m, inj_m = monthly_volumes(res, Q_inj)
    n = min(len(oil_m), int(econ.max_life_years * 12))
    oil_m, watp_m, inj_m = oil_m[:n], watp_m[:n], inj_m[:n]

    revenue = oil_m * econ.oil_price * econ.nri
    opex = (econ.opex_fixed
            + oil_m * econ.opex_oil
            + (watp_m + inj_m) * econ.water_cost)
    op_cf = revenue - opex

    # economic limit: truncate at the first month with negative operating CF
    neg = np.where(op_cf < 0)[0]
    n_life = int(neg[0]) if len(neg) else n
    if n_life == 0:
        n_life = 1                                     # degenerate guard
    op_cf = op_cf[:n_life]
    oil_m = oil_m[:n_life]

    months = np.arange(1, n_life + 1)
    disc = (1.0 + econ.disc_rate) ** (-(months - 0.5) / 12.0)  # mid-month
    capex = econ.capex(Q_inj)
    npv = float(np.sum(op_cf * disc) - capex)

    cum_cf = np.cumsum(op_cf) - capex                  # undiscounted, for payout
    pay = np.where(cum_cf > 0)[0]
    payout_months = float(pay[0] + 1) if len(pay) else np.nan

    return {
        'Q_inj': Q_inj,
        'NPV': npv,
        'IRR': irr_annual(capex, op_cf),
        'payout_months': payout_months,
        'EUR_bbl': float(oil_m.sum()),
        'RF_at_limit': float(oil_m.sum() / res['OOIP_movable']),
        'life_years': n_life / 12.0,
        'capex': capex,
        'op_cf': op_cf,
    }


def irr_annual(capex, op_cf, lo=-0.99, hi=10.0, tol=1e-6):
    """Annualized IRR by bisection on the monthly cash flow NPV."""
    months = np.arange(1, len(op_cf) + 1)

    def npv_at(r_annual):
        d = (1.0 + r_annual) ** (-(months - 0.5) / 12.0)
        return np.sum(op_cf * d) - capex

    if npv_at(lo) * npv_at(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv_at(lo) * npv_at(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 3. RATE OPTIMIZATION
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    econ = EconInputs()

    # one physics run, deliberately carried well past breakthrough so the
    # economic limit (not the simulation end) is what truncates production
    cfg = Config(PVI_end=2.5, snap_PVI=())
    print("Running physics once (rate-independent in PVI space)...")
    res = Simulator(cfg).run(verbose=False)
    print(f"  breakthrough at {res['breakthrough_PVI']:.3f} PVI, "
          f"mass balance error {res['mass_balance_error']:+.1e}\n")

    rates = np.array([150, 250, 350, 500, 750, 1000, 1500, 2000, 3000], float)
    evals = [evaluate(res, Q, econ) for Q in rates]

    hdr = (f"{'Q_inj (bbl/d)':>13}{'NPV10 ($MM)':>13}{'IRR':>8}"
           f"{'payout (mo)':>13}{'EUR (Mbbl)':>12}{'RF @ limit':>12}"
           f"{'life (yr)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for e in evals:
        print(f"{e['Q_inj']:>13.0f}{e['NPV']/1e6:>13.2f}{e['IRR']:>8.1%}"
              f"{e['payout_months']:>13.0f}{e['EUR_bbl']/1e3:>12.0f}"
              f"{e['RF_at_limit']:>12.3f}{e['life_years']:>11.1f}")

    best_npv = max(evals, key=lambda e: e['NPV'])
    best_eur = max(evals, key=lambda e: e['EUR_bbl'])
    print(f"\nNPV-optimal rate : {best_npv['Q_inj']:.0f} bbl/d "
          f"(NPV ${best_npv['NPV']/1e6:.2f}MM, "
          f"EUR {best_npv['EUR_bbl']/1e3:.0f} Mbbl)")
    print(f"EUR-optimal rate : {best_eur['Q_inj']:.0f} bbl/d "
          f"(NPV ${best_eur['NPV']/1e6:.2f}MM, "
          f"EUR {best_eur['EUR_bbl']/1e3:.0f} Mbbl)")
    dnpv = best_npv['NPV'] - best_eur['NPV']
    print(f"Chasing maximum barrels instead of value would leave "
          f"${dnpv/1e6:.2f}MM of NPV on the table.")

    # ---- figure: the recovery-vs-value trade-off ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.plot(rates, [e['NPV'] / 1e6 for e in evals], 'o-', color='tab:green', lw=2)
    ax.axvline(best_npv['Q_inj'], color='tab:green', ls='--', lw=1)
    ax.annotate(f"NPV max\n{best_npv['Q_inj']:.0f} bbl/d",
                (best_npv['Q_inj'], best_npv['NPV'] / 1e6),
                textcoords="offset points", xytext=(10, -30), fontsize=9)
    ax.set_xlabel("injection rate (bbl/d)")
    ax.set_ylabel("NPV10 ($MM)")
    ax.set_title("Value: interior optimum")

    ax = axes[1]
    ax.plot(rates, [e['EUR_bbl'] / 1e3 for e in evals], 's-',
            color='tab:brown', lw=2)
    ax.axvline(best_eur['Q_inj'], color='tab:brown', ls='--', lw=1)
    ax.set_xlabel("injection rate (bbl/d)")
    ax.set_ylabel("EUR to economic limit (Mbbl)")
    ax.set_title("Barrels: keeps rising")

    ax = axes[2]
    e = best_npv
    months = np.arange(1, len(e['op_cf']) + 1)
    ax.bar(months / 12.0, e['op_cf'] / 1e3, width=1 / 12.0,
           color='tab:blue', alpha=0.7, label="operating CF ($M/mo)")
    disc = (1.0 + econ.disc_rate) ** (-(months - 0.5) / 12.0)
    cum_disc = (np.cumsum(e['op_cf'] * disc) - e['capex']) / 1e6
    axb = ax.twinx()
    axb.plot(months / 12.0, cum_disc, 'r-', lw=2,
             label="cumulative discounted CF ($MM)")
    axb.axhline(0, color='k', lw=0.8)
    axb.set_ylabel("cumulative discounted CF ($MM)", color='r')
    axb.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel("years")
    ax.set_ylabel("monthly operating CF ($M)", color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title(f"Cash flow profile at NPV-optimal rate "
                 f"({best_npv['Q_inj']:.0f} bbl/d)")

    fig.suptitle("Optimizing on NPV, not recovery: more oil is not more value",
                 fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig("figures/npv_optimization.png", dpi=130, bbox_inches='tight')
    print("Saved figures/npv_optimization.png")
