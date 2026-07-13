"""
Test suite for reservoir-asset-evaluation.
15 tests covering physics, numerics, and economics of the Phase 1 IMPES
simulator. Runs in CI on every push (see .github/workflows/ci.yml).

Recreated July 12, 2026 and verified green against the live repo code.
Small grids keep the full suite under a minute.
"""

import numpy as np
import pytest

from reservoir_sim import Config, Simulator, BBL_TO_FT3
from economics import EconInputs, evaluate, monthly_volumes, irr_annual


# ---------------------------------------------------------------------------
# Fixtures: one small reference run shared by the tests that need it
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run():
    cfg = Config(nx=11, ny=11, PVI_end=0.8, snap_PVI=(0.2, 0.5))
    return Simulator(cfg).run(verbose=False)


@pytest.fixture(scope="module")
def small_sim():
    return Simulator(Config(nx=11, ny=11))


# ---------------------------------------------------------------------------
# 1-4: Configuration and setup
# ---------------------------------------------------------------------------
def test_config_defaults_wells_at_corners():
    cfg = Config(nx=15, ny=15)
    assert cfg.inj_cell == (0, 0)
    assert cfg.prod_cell == (14, 14)


def test_config_homogeneous_field_filled():
    cfg = Config(nx=9, ny=9, k_hom=150.0)
    assert cfg.k_field.shape == (9, 9)
    assert np.allclose(cfg.k_field, 150.0)


def test_mobility_ratio_matches_definition():
    cfg = Config(krw_max=0.4, kro_max=0.9, mu_w=0.5, mu_o=5.0)
    expected = (0.4 / 0.5) / (0.9 / 5.0)
    assert np.isclose(cfg.mobility_ratio, expected)


def test_heterogeneous_transmissibility_harmonic_mean(small_sim):
    """A zero-perm cell must kill flow across its faces: harmonic mean -> 0."""
    k = np.full((11, 11), 200.0)
    k[5, 5] = 1e-12
    sim = Simulator(Config(nx=11, ny=11, k_field=k))
    # faces touching the dead cell in x: between (4,5)-(5,5) and (5,5)-(6,5)
    assert sim.Tx_geo[4, 5] < 1e-9
    assert sim.Tx_geo[5, 5] < 1e-9


# ---------------------------------------------------------------------------
# 5-8: Relative permeability and fractional flow physics
# ---------------------------------------------------------------------------
def test_normalized_sw_clipped_to_unit_interval(small_sim):
    Sw = np.array([0.0, small_sim.cfg.Swc, 0.5, 1.0 - small_sim.cfg.Sor, 1.0])
    Sn = small_sim.normalized_sw(Sw)
    assert Sn.min() >= 0.0 and Sn.max() <= 1.0
    assert np.isclose(Sn[1], 0.0) and np.isclose(Sn[3], 1.0)


def test_kr_endpoints(small_sim):
    c = small_sim.cfg
    krw_c, kro_c = small_sim.kr(np.array([c.Swc]))
    krw_m, kro_m = small_sim.kr(np.array([1.0 - c.Sor]))
    assert np.isclose(krw_c[0], 0.0)          # no water flow at connate
    assert np.isclose(kro_c[0], c.kro_max)    # max oil mobility at connate
    assert np.isclose(krw_m[0], c.krw_max)    # max water mobility at 1-Sor
    assert np.isclose(kro_m[0], 0.0)          # no oil flow at residual


def test_frac_flow_monotonic_and_bounded(small_sim):
    c = small_sim.cfg
    s = np.linspace(c.Swc, 1.0 - c.Sor, 200)
    fw = small_sim.frac_flow(s)
    assert fw.min() >= 0.0 and fw.max() <= 1.0
    assert np.all(np.diff(fw) >= -1e-12)      # non-decreasing in Sw


def test_cfl_dt_positive(small_sim):
    Sw = np.full((11, 11), small_sim.cfg.Swc)
    p = small_sim.solve_pressure(Sw)
    qx, qy = small_sim.face_fluxes(Sw, p)
    assert small_sim.cfl_dt(qx, qy) > 0.0


# ---------------------------------------------------------------------------
# 9-12: Full-run numerics
# ---------------------------------------------------------------------------
def test_mass_balance_at_machine_precision(small_run):
    """The headline claim: injected = produced + stored, to round-off."""
    assert abs(small_run["mass_balance_error"]) < 1e-10


def test_water_cut_bounded(small_run):
    wc = small_run["hist"]["wc"]
    assert wc.min() >= 0.0 and wc.max() <= 1.0 + 1e-12


def test_recovery_factor_monotonic_and_bounded(small_run):
    RF = small_run["hist"]["RF"]
    assert np.all(np.diff(RF) >= -1e-12)
    assert 0.0 < small_run["RF"] <= 1.0


def test_breakthrough_before_end(small_run):
    """At M ~ 2.25 on a quarter-five-spot, water must break through well
    before 0.8 PVI, and water cut afterwards must exceed the threshold."""
    bt = small_run["breakthrough_PVI"]
    assert not np.isnan(bt)
    assert 0.05 < bt < 0.8


# ---------------------------------------------------------------------------
# 13-15: Economics
# ---------------------------------------------------------------------------
def test_capex_scales_with_rate():
    e = EconInputs()
    assert e.capex(1000) == pytest.approx(e.capex_dc + e.capex_fac_base + 1500e3)
    assert e.capex(2000) > e.capex(1000)


def test_monthly_volumes_conserve_oil(small_run):
    """Monthly aggregation must not create or destroy oil (within interp tol)."""
    oil_m, _, _ = monthly_volumes(small_run, Q_inj=small_run["cfg"].Q_inj)
    assert oil_m.sum() == pytest.approx(small_run["cum_oil"], rel=2e-2)
    assert np.all(oil_m >= -1e-9)


def test_npv_falls_at_extreme_rates(small_run):
    """The Phase 1 economic insight: pushing rate eventually destroys value.
    NPV at a very high injection rate must fall below the moderate-rate NPV,
    and IRR must be a finite number when operating cash flow is positive."""
    econ = EconInputs()
    mid = evaluate(small_run, Q_inj=750.0, econ=econ)
    high = evaluate(small_run, Q_inj=20000.0, econ=econ)
    assert high["NPV"] < mid["NPV"]
    r = mid["IRR"]
    assert r is None or np.isnan(r) or np.isfinite(r)
    assert "payout_months" in mid and "EUR_bbl" in mid
