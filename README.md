# Reservoir Asset Evaluation Engine

I am a petroleum engineering student at the University of Texas at Austin working toward becoming a reservoir engineer. I decided to build this simulator rather than just read a textbook, to learn the physics and economics of a well while having a little fun. It was, and it only furthered my interest in the energy sector.

The project does the full loop a reserves evaluator runs on a producing asset: simulate the physics, turn the production into cash flow, and find the design that makes the most money instead of the most barrels.

**The physics.** I wrote a 2D two-phase (oil and water) simulator in Python using the IMPES method: solve pressure implicitly with a sparse matrix, then move water saturation explicitly with upstream weighting. It handles any permeability field you give it, uses Corey relative permeability curves, and picks its own stable time step from the CFL condition. Every run prints a water mass balance check, and it comes back at machine precision (about 1e-14), so mass is conserved by construction, not by luck.

**Proving it works.** A simulator is only worth trusting if it can reproduce a known answer. The 1D Buckley-Leverett problem has an exact analytical solution, so I ran my simulator against it. The saturation profiles land on top of the analytical solution, breakthrough timing and recovery agree within 1.2% on a 200-cell grid, and refining the grid sharpens the front toward the exact answer.

![Buckley-Leverett validation](figures/validation_buckley_leverett.png)

**The economics.** The production stream feeds a monthly discounted cash flow model: flat oil price, 25% royalty, fixed and variable opex, water handling costs, and facilities capex that scales with injection rate. Production stops at the economic limit, the first month cash flow goes negative. Sweeping the injection rate shows the whole point of the project: NPV peaks at 750 bbl/d, but recovered barrels keep climbing all the way to 3,000 bbl/d. Chasing maximum barrels instead of maximum value gives up about $2.2MM on a $5.6MM asset. More oil is not more value.

![NPV optimization](figures/npv_optimization.png)

## Key results

| Study | What I found |
|---|---|
| Validation (1D, 200 cells) | Breakthrough and recovery within 1.2% of the analytical solution |
| Mobility ratio (M = 0.19 to 3.89) | Recovery at 1 PVI falls from 0.97 to 0.74 as M rises |
| Heterogeneity (same 200 mD average) | A connected high-perm channel cuts recovery from 0.84 to 0.51 |
| NPV optimization | Value peaks at 750 bbl/d; barrels peak at the highest rate tested |

![Heterogeneity](figures/heterogeneity_fields.png)

## What is in here

```
reservoir_sim.py               the core simulator, run it directly for the base case
validate_buckley_leverett.py   1D check against the analytical solution
mobility_ratio_study.py        four oil viscosities, sweep and recovery comparison
heterogeneity_study.py         channel and random permeability fields
economics.py                   DCF model, economic limit, NPV vs injection rate
writeup/simulator_writeup.md   the full math, validation, results, and limitations
figures/                       all output figures
```

## Running it

```
pip install -r requirements.txt
python reservoir_sim.py
python validate_buckley_leverett.py
python mobility_ratio_study.py
python heterogeneity_study.py
python economics.py
```

Each script prints a summary table (including the mass balance check) and saves its figures to `figures/`.

## Being upfront about scope

I built everything here from first principles in Python (NumPy, SciPy, Matplotlib), with AI as a coding assistant along the way. I have not used commercial tools like ARIES, PHDWin, or ComboCurve hands-on; this project is my way of understanding the methods those tools run under the hood. The model leaves things out on purpose (no compressibility, gravity, or capillary pressure, a simple well model, first-order transport), and the writeup covers each one with the reasoning and the standard fix.

## What is next

- Monte Carlo over permeability, oil in place, and price for P10/P50/P90 reserves and value
- A proper Peaceman well model and multi-well patterns
- A decline curve module: fit Arps curves to my own simulated production and see how close the estimated EUR gets to the true answer
- An interactive dashboard for sensitivity analysis
