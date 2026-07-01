"""
2D Two-Phase (oil/water) Finite-Volume Reservoir Simulator - IMPES method
=========================================================================

Immiscible, incompressible water-oil flow on a rectangular grid using IMPES
(IMplicit Pressure, Explicit Saturation):

    1. Solve the PRESSURE equation implicitly (one sparse linear system) using
       the total mobility evaluated at the current saturation.
    2. Compute the Darcy flux across every cell face from that pressure field.
    3. Advance WATER SATURATION explicitly with upstream (upwind) weighting of
       the fractional flow.

Base setup: quarter five-spot waterflood. Water injected at one corner, total
fluid produced at the opposite corner (held at fixed pressure). Field units
throughout (mD, cp, ft, psi, bbl/day). Darcy constant beta = 1.127e-3.

This module is the single source of truth for the physics. The study scripts
(validation, mobility ratio, heterogeneity, economics) import it rather than
duplicating code:

    from reservoir_sim import Config, Simulator

    cfg = Config(mu_o=10.0, PVI_end=1.5)
    result = Simulator(cfg).run()

Every run reports a water material balance error so the numerics can be
checked at a glance, not taken on faith.

Author: Hatton (UT Austin, Petroleum Engineering)
"""

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BETA = 1.127e-3      # Darcy constant, field units
BBL_TO_FT3 = 5.615   # barrels -> cubic feet


# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
@dataclass
class Config:
    """All physical and numerical inputs. Change these to experiment."""
    # --- grid / geometry ---
    nx: int = 31
    ny: int = 31
    Lx: float = 1000.0          # reservoir extent, ft
    Ly: float = 1000.0
    h: float = 20.0             # thickness, ft

    # --- rock ---
    phi: float = 0.20           # porosity (fraction)
    k_field: np.ndarray = None  # permeability array (nx, ny) in mD;
                                # None -> homogeneous k_hom everywhere
    k_hom: float = 200.0        # homogeneous permeability, mD

    # --- fluids (Corey relative permeability) ---
    mu_w: float = 1.0           # water viscosity, cp
    mu_o: float = 4.0           # oil viscosity, cp
    Swc: float = 0.20           # connate water saturation
    Sor: float = 0.20           # residual oil saturation
    krw_max: float = 0.35
    kro_max: float = 0.90
    nw: float = 2.0             # Corey exponents
    no: float = 2.0

    # --- wells ---
    inj_cell: tuple = None      # default (0, 0)
    prod_cell: tuple = None     # default (nx-1, ny-1)
    Q_inj: float = 500.0        # water injection rate, bbl/day
    p_ref: float = 1000.0       # producer bottomhole pressure, psi (Dirichlet)

    # --- run control ---
    PVI_end: float = 1.0        # stop after this many pore volumes injected
    cfl: float = 0.5            # stability safety factor for the explicit step
    snap_PVI: tuple = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)  # saturation snapshots
    max_steps: int = 60000

    def __post_init__(self):
        if self.inj_cell is None:
            self.inj_cell = (0, 0)
        if self.prod_cell is None:
            self.prod_cell = (self.nx - 1, self.ny - 1)
        if self.k_field is None:
            self.k_field = np.full((self.nx, self.ny), self.k_hom)
        assert self.k_field.shape == (self.nx, self.ny), "k_field must be (nx, ny)"

    @property
    def mobility_ratio(self):
        """End-point mobility ratio M = (krw_max/mu_w) / (kro_max/mu_o)."""
        return (self.krw_max / self.mu_w) / (self.kro_max / self.mu_o)


# ----------------------------------------------------------------------------
# 2. SIMULATOR
# ----------------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        c = cfg
        self.dx = c.Lx / c.nx
        self.dy = c.Ly / c.ny
        self.N = c.nx * c.ny
        self.Vcell = self.dx * self.dy * c.h                # bulk cell volume, ft3
        self.PV_total = c.phi * c.Lx * c.Ly * c.h           # total pore volume, ft3
        # movable oil in place, bbl
        self.OOIP_movable = (1.0 - c.Swc - c.Sor) * self.PV_total / BBL_TO_FT3

        # linear indices of the wells: idx(i, j) = j*nx + i
        self.inj_lin = c.inj_cell[1] * c.nx + c.inj_cell[0]
        self.prod_lin = c.prod_cell[1] * c.nx + c.prod_cell[0]

        # geometric (mobility-free) face transmissibilities.
        # Harmonic mean of neighboring permeabilities: flow across a face passes
        # through both cells in series, so the tighter rock limits it.
        k = c.k_field
        kx = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])   # (nx-1, ny)
        self.Tx_geo = BETA * kx * (self.dy * c.h) / self.dx
        if c.ny > 1:
            ky = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])  # (nx, ny-1)
            self.Ty_geo = BETA * ky * (self.dx * c.h) / self.dy
        else:
            self.Ty_geo = np.zeros((c.nx, 0))

        # max slope of the fractional flow curve, used by the CFL condition.
        # It depends only on fluid properties, so compute it once, not per step.
        s = np.linspace(c.Swc, 1.0 - c.Sor, 400)
        self.dfw_max = np.max(np.abs(np.gradient(self.frac_flow(s), s)))

    # --- relative permeability / fractional flow (Corey) -------------------
    def normalized_sw(self, Sw):
        c = self.cfg
        return np.clip((Sw - c.Swc) / (1.0 - c.Swc - c.Sor), 0.0, 1.0)

    def kr(self, Sw):
        c = self.cfg
        Sn = self.normalized_sw(Sw)
        return c.krw_max * Sn ** c.nw, c.kro_max * (1.0 - Sn) ** c.no

    def mobilities(self, Sw):
        krw, kro = self.kr(Sw)
        return krw / self.cfg.mu_w, kro / self.cfg.mu_o

    def frac_flow(self, Sw):
        lw, lo = self.mobilities(Sw)
        lt = lw + lo
        return np.where(lt > 0, lw / lt, 0.0)

    # --- face transmissibilities at current saturation ----------------------
    def face_transmissibilities(self, Sw):
        """Geometric transmissibility times face total mobility (arithmetic avg)."""
        lw, lo = self.mobilities(Sw)
        lt = lw + lo
        Tx = self.Tx_geo * 0.5 * (lt[:-1, :] + lt[1:, :])
        Ty = (self.Ty_geo * 0.5 * (lt[:, :-1] + lt[:, 1:])
              if self.cfg.ny > 1 else self.Ty_geo)
        return Tx, Ty

    # --- pressure solve (implicit) ------------------------------------------
    def solve_pressure(self, Sw):
        c = self.cfg
        nx, ny, N = c.nx, c.ny, self.N
        Tx, Ty = self.face_transmissibilities(Sw)

        rows, cols, data = [], [], []
        diag = np.zeros(N)

        # horizontal connections
        I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny), indexing='ij')
        cc = (J * nx + I).ravel()
        cn = cc + 1
        T = Tx.ravel()
        rows += [cc, cn]; cols += [cn, cc]; data += [T, T]
        np.add.at(diag, cc, -T); np.add.at(diag, cn, -T)

        # vertical connections
        if ny > 1:
            I, J = np.meshgrid(np.arange(nx), np.arange(ny - 1), indexing='ij')
            cc = (J * nx + I).ravel()
            cn = cc + nx
            T = Ty.ravel()
            rows += [cc, cn]; cols += [cn, cc]; data += [T, T]
            np.add.at(diag, cc, -T); np.add.at(diag, cn, -T)

        rows.append(np.arange(N)); cols.append(np.arange(N)); data.append(diag)
        A = sp.csr_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N))

        b = np.zeros(N)
        b[self.inj_lin] = -c.Q_inj          # rate source (injection)

        # Dirichlet at producer: replace its row with identity.
        # One fixed pressure is required; with both wells on rate control the
        # incompressible system only defines pressure differences (singular A).
        A = A.tolil()
        A.rows[self.prod_lin] = [self.prod_lin]
        A.data[self.prod_lin] = [1.0]
        A = A.tocsr()
        b[self.prod_lin] = c.p_ref

        return spla.spsolve(A, b)

    # --- fluxes from a pressure field ---------------------------------------
    def face_fluxes(self, Sw, p):
        """Signed total fluxes, positive in the +i / +j direction, bbl/day."""
        c = self.cfg
        Tx, Ty = self.face_transmissibilities(Sw)
        Pg = p.reshape(c.ny, c.nx).T                 # P[i, j]
        qx = Tx * (Pg[:-1, :] - Pg[1:, :])           # (nx-1, ny)
        qy = (Ty * (Pg[:, :-1] - Pg[:, 1:])
              if c.ny > 1 else np.zeros((c.nx, 0)))  # (nx, ny-1)
        return qx, qy

    def _inflow_to_producer(self, qx, qy):
        """Total rate flowing into the producer cell (bbl/day), any geometry."""
        c = self.cfg
        ip, jp = c.prod_cell
        q = 0.0
        if ip > 0:
            q += max(qx[ip - 1, jp], 0.0)            # from west neighbor
        if ip < c.nx - 1:
            q += max(-qx[ip, jp], 0.0)               # from east neighbor
        if c.ny > 1 and jp > 0:
            q += max(qy[ip, jp - 1], 0.0)            # from south neighbor
        if c.ny > 1 and jp < c.ny - 1:
            q += max(-qy[ip, jp], 0.0)               # from north neighbor
        return q

    # --- explicit saturation update with upstream weighting -----------------
    def step(self, Sw, qx, qy, dt):
        c = self.cfg
        fw = self.frac_flow(Sw)

        # upstream fractional flow on each face: the fluid crossing a face
        # carries the composition of the cell it came from.
        fwx = np.where(qx >= 0, fw[:-1, :], fw[1:, :])
        qwx = fwx * qx
        if c.ny > 1:
            fwy = np.where(qy >= 0, fw[:, :-1], fw[:, 1:])
            qwy = fwy * qy

        qw_net = np.zeros((c.nx, c.ny))              # net water in, bbl/day
        qw_net[:-1, :] -= qwx; qw_net[1:, :] += qwx
        if c.ny > 1:
            qw_net[:, :-1] -= qwy; qw_net[:, 1:] += qwy

        qt_prod = self._inflow_to_producer(qx, qy)
        fw_prod = float(self.frac_flow(np.asarray(Sw[c.prod_cell])))

        qw_net[c.inj_cell] += c.Q_inj                # injector: pure water in
        qw_net[c.prod_cell] -= fw_prod * qt_prod     # producer: water out at its fw

        dSw = dt * BBL_TO_FT3 * qw_net / (c.phi * self.Vcell)
        Sw_new = np.clip(Sw + dSw, c.Swc, 1.0 - c.Sor)

        oil_prod = (1.0 - fw_prod) * qt_prod * dt    # bbl this step
        wat_prod = fw_prod * qt_prod * dt
        return Sw_new, fw_prod, qt_prod, oil_prod, wat_prod

    def cfl_dt(self, qx, qy):
        """Stable explicit time step (days) from the CFL condition."""
        maxflux = max(np.abs(qx).max() if qx.size else 0.0,
                      np.abs(qy).max() if qy.size else 0.0, 1e-9)
        return self.cfg.cfl * (self.cfg.phi * self.Vcell) / (
            BBL_TO_FT3 * maxflux * self.dfw_max)

    # --- main time loop ------------------------------------------------------
    def run(self, verbose=True):
        c = self.cfg
        Sw = np.full((c.nx, c.ny), c.Swc)            # reservoir starts full of oil
        t = 0.0
        cum_inj = cum_oil = cum_wat = 0.0

        snap_targets = sorted(c.snap_PVI)
        snaps = {}
        hist = {'t': [], 'dt': [], 'PVI': [], 'wc': [], 'RF': [],
                'oil_rate': [], 'wat_rate': [], 'qt_prod': []}

        steps = 0
        while True:
            p = self.solve_pressure(Sw)
            qx, qy = self.face_fluxes(Sw, p)
            dt = self.cfl_dt(qx, qy)
            Sw, fw_prod, qt_prod, oil_p, wat_p = self.step(Sw, qx, qy, dt)

            t += dt
            cum_inj += c.Q_inj * dt
            cum_oil += oil_p
            cum_wat += wat_p
            PVI = cum_inj * BBL_TO_FT3 / self.PV_total

            hist['t'].append(t)
            hist['dt'].append(dt)
            hist['PVI'].append(PVI)
            hist['wc'].append(fw_prod)
            hist['RF'].append(cum_oil / self.OOIP_movable)
            hist['oil_rate'].append(oil_p / dt)
            hist['wat_rate'].append(wat_p / dt)
            hist['qt_prod'].append(qt_prod)

            while snap_targets and PVI >= snap_targets[0]:
                snaps[snap_targets.pop(0)] = Sw.copy()

            steps += 1
            if PVI >= c.PVI_end or steps >= c.max_steps:
                break

        hist = {key: np.asarray(v) for key, v in hist.items()}

        # water material balance: injected = produced + change in stored water.
        # This is the sanity check that the discrete scheme conserves mass.
        stored = np.sum((Sw - c.Swc)) * c.phi * self.Vcell / BBL_TO_FT3   # bbl
        mb_err = (cum_inj - cum_wat - stored) / max(cum_inj, 1e-12)

        result = {
            'cfg': c, 'Sw': Sw, 'p': p, 'snaps': snaps, 'hist': hist,
            'steps': steps, 'PVI': PVI,
            'cum_oil': cum_oil, 'cum_wat': cum_wat, 'cum_inj': cum_inj,
            'RF': cum_oil / self.OOIP_movable,
            'OOIP_movable': self.OOIP_movable,
            'PV_total': self.PV_total,
            'wc_final': fw_prod,
            'mass_balance_error': mb_err,
            'breakthrough_PVI': self.breakthrough_PVI(hist),
        }
        if verbose:
            print(f"steps={steps}  PVI={PVI:.3f}  M={c.mobility_ratio:.2f}  "
                  f"breakthrough PVI={result['breakthrough_PVI']:.3f}  "
                  f"water cut={fw_prod:.3f}  RF(movable)={result['RF']:.3f}")
            print(f"movable OOIP={self.OOIP_movable:,.0f} bbl  "
                  f"cum oil={cum_oil:,.0f} bbl  "
                  f"water mass balance error={mb_err:+.2e}")
        return result

    @staticmethod
    def breakthrough_PVI(hist, threshold=0.01):
        """First PVI at which the producer water cut exceeds the threshold."""
        idx = np.argmax(hist['wc'] > threshold)
        if hist['wc'][idx] <= threshold:
            return np.nan                             # never broke through
        return hist['PVI'][idx]


# ----------------------------------------------------------------------------
# 3. BASE CASE (run this file directly)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    os.makedirs("figures", exist_ok=True)
    cfg = Config()
    res = Simulator(cfg).run()
    snaps, hist = res['snaps'], res['hist']
    extent = [0, cfg.Lx, 0, cfg.Ly]

    # (a) saturation evolution
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, target in zip(axes.ravel(), sorted(snaps.keys())):
        im = ax.imshow(snaps[target].T, origin='lower', extent=extent,
                       vmin=cfg.Swc, vmax=1 - cfg.Sor, cmap='turbo', aspect='equal')
        ax.set_title(f"Water saturation @ {target:.1f} PVI")
        ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="$S_w$")
    fig.suptitle("Quarter five-spot waterflood: saturation front advance",
                 fontsize=14, weight='bold')
    fig.savefig("figures/saturation_evolution.png", dpi=130, bbox_inches='tight')

    # (b) final pressure field
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.2))
    im2 = ax2.imshow(res['p'].reshape(cfg.ny, cfg.nx), origin='lower',
                     extent=extent, cmap='viridis', aspect='equal')
    ax2.set_title("Pressure field (psi) at final time")
    ax2.set_xlabel("x (ft)"); ax2.set_ylabel("y (ft)")
    fig2.colorbar(im2, ax=ax2, label="pressure (psi)")
    fig2.savefig("figures/pressure_field.png", dpi=130, bbox_inches='tight')

    # (c) water cut & recovery vs PVI
    fig3, ax3 = plt.subplots(figsize=(7, 4.6))
    ax3.plot(hist['PVI'], hist['wc'], 'b-', lw=2)
    ax3.set_xlabel("Pore volumes injected (PVI)")
    ax3.set_ylabel("Water cut  $f_w$", color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_ylim(0, 1)
    ax3b = ax3.twinx()
    ax3b.plot(hist['PVI'], hist['RF'], 'r-', lw=2)
    ax3b.set_ylabel("Recovery factor (movable)", color='r')
    ax3b.tick_params(axis='y', labelcolor='r')
    ax3b.set_ylim(0, 1)
    ax3.set_title("Breakthrough curve & oil recovery")
    fig3.savefig("figures/breakthrough_recovery.png", dpi=130, bbox_inches='tight')

    print("Saved figures to figures/.")
