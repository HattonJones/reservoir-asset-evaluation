"""
Phase 2 tests: Haynesville decline curve analysis engine.
Validates the Arps fitting, terminal-decline switch, and PDQ loader so the
CI badge covers both phases.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase2"))
from haynesville_decline import (
    arps_hyperbolic, fit_models, rate_with_terminal, load_pdq_csv, DAYS_PER_MONTH,
)


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    """60-month synthetic gas well with known parameters + 5% lognormal noise
    (the same validation protocol the engine was built against)."""
    rng = np.random.default_rng(42)
    qi, Di_ann, b = 12000.0, 0.85, 1.10
    Di_m = -np.log(1 - Di_ann) / 12
    t = np.arange(60)
    q = qi / (1 + b * Di_m * t) ** (1 / b)
    q_noisy = q * rng.lognormal(0, 0.05, 60)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    p = tmp_path_factory.mktemp("dca") / "synth.csv"
    pd.DataFrame({
        "Cycle Year-Month": dates.strftime("%Y-%m"),
        "Total Gas (MCF)": (q_noisy * DAYS_PER_MONTH).round(0),
    }).to_csv(p, index=False)
    return p, dict(qi=qi, b=b)


def test_loader_detects_pdq_columns(synthetic_csv):
    path, _ = synthetic_csv
    df = load_pdq_csv(path)
    assert list(df.columns) == ["date", "gas_mcf", "rate_mcfd"]
    assert len(df) == 60
    assert (df["rate_mcfd"] > 0).all()


def test_hyperbolic_recovers_known_parameters(synthetic_csv):
    path, truth = synthetic_csv
    df = load_pdq_csv(path)
    t = np.arange(len(df), dtype=float)
    q = df["rate_mcfd"].to_numpy()
    res = fit_models(t, q)
    best = max(res, key=lambda k: res[k]["r2"])
    assert best == "hyperbolic"
    p = res["hyperbolic"]["params"]
    assert p["qi"] == pytest.approx(truth["qi"], rel=0.05)   # qi within 5%
    assert p["b"] == pytest.approx(truth["b"], abs=0.15)     # b near truth


def test_terminal_switch_bounds_reserves():
    """A b >= 1 hyperbolic has unbounded EUR; the terminal switch must make
    the tail integrable (exponential), so late-time rates fall much faster
    than the pure hyperbolic."""
    qi, Di, b = 10000.0, 0.15, 1.2
    dmin = -np.log(1 - 0.06) / 12
    t_late = np.array([600.0, 1200.0])                        # 50 and 100 years
    q_mod = rate_with_terminal(t_late, qi, Di, b, dmin)
    q_pure = arps_hyperbolic(t_late, qi, Di, b)
    assert (q_mod < q_pure).all()
    # exponential tail: ratio over the second 50 years matches exp decay
    assert q_mod[1] / q_mod[0] == pytest.approx(np.exp(-dmin * 600.0), rel=1e-6)


def test_rate_with_terminal_continuous_at_switch():
    qi, Di, b = 20000.0, 0.10, 0.73
    dmin = -np.log(1 - 0.06) / 12
    t_sw = (Di / dmin - 1.0) / (b * Di)
    eps = 1e-6
    q = rate_with_terminal(np.array([t_sw - eps, t_sw + eps]), qi, Di, b, dmin)
    assert q[0] == pytest.approx(q[1], rel=1e-4)
