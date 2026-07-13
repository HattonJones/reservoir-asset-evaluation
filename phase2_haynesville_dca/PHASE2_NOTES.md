# Phase 2: Haynesville Decline Curve Analysis

Single-well decline curve analysis and EUR estimate for a producing Haynesville
gas well, using free public monthly production data from the Texas Railroad
Commission Production Data Query (PDQ).

## The well

| | |
|---|---|
| Well | Harrison LH A 1H |
| Operator | Comstock Resources |
| API | 42-203-35464 |
| District | Texas RRC District 06 |
| Play | Haynesville Shale (East Texas) |

## Results

| | |
|---|---|
| Best-fit model | Arps hyperbolic |
| qi | 23,871 Mcf/d |
| b-factor | 0.73 |
| Terminal decline switch | 6%/yr |
| Economic limit | 100 Mcf/d |
| Cum to date | 12.12 Bcf (49 months) |
| EUR | 20.59 Bcf |

## What the script does

`haynesville_decline.py`:

1. Loads the PDQ CSV and auto-detects the date and gas columns. Drops shut-in
   months at the tail and converts monthly Mcf to Mcf/day.
2. Fits three Arps models (exponential, hyperbolic, harmonic) in log space,
   starting from the peak-rate month. Fitting in log space keeps the late-time
   data from being ignored, since late rates are small in absolute terms but
   matter most for EUR.
3. Applies a terminal decline switch. A hyperbolic curve with b above 1 has an
   unbounded EUR if you let it run forever. The script switches to exponential
   decline (default 6% per year) once the instantaneous decline rate falls to
   that level. This is standard reserves practice.
4. Forecasts to an economic limit rate (default 100 Mcf/d) and reports EUR in Bcf.
5. Outputs a summary text file and two plots: rate vs. time on a semilog axis,
   and cumulative production with the EUR line.

## Usage

```
pip install -r requirements.txt
python haynesville_decline.py production.csv --name "WELL NAME 1H" --api 42-203-XXXXX
```

Options: `--econ-limit` (Mcf/d, default 100) and `--terminal` (annual decline
fraction, default 0.06).

## Validation

The engine was tested on a synthetic 60-month gas well generated with known
parameters (qi = 12,000 Mcf/d, Di = 85% per year nominal, b = 1.10, 5%
lognormal noise). The script correctly selected the hyperbolic model and
matched the rate profile at the noise level. The recovered qi was within 5% of
truth and b was recovered exactly. Di and b trade off against each other in
Arps fitting, which is a known behavior, so individual parameters can differ
from truth even when the curve and EUR are sound.

## Limitations

This is single-well analysis on public monthly data, and I want to be honest
about what it cannot see:

- Monthly volumes smooth out early-time behavior. Daily data would fit the
  first year better.
- The script cannot tell downtime from true decline. A workover month looks
  like a decline point.
- Choke management is common on Haynesville wells. A well held flat on choke
  violates the Arps assumption of constant flowing conditions, so the fit
  should start after the choke period ends, not just at the peak month.
- b-factors are sensitive to how much early data you exclude. Small changes in
  the fit window can move EUR meaningfully.
- No economics layer yet. Adding a gas price deck and NPV10 calculation on top
  of the forecast is a natural next step.
