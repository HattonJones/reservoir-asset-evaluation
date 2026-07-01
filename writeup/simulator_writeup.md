# 2D Two-Phase Reservoir Simulator: Physics, Validation, and Asset Economics

A finite-volume, IMPES (Implicit Pressure, Explicit Saturation) simulator for
incompressible oil-water flow in a horizontal reservoir, validated against the
Buckley-Leverett analytical solution and coupled to a discounted cash flow model
that optimizes flood design on NPV. This document states the physics, the
discretization, and the numerical choices, with the reasoning behind each one,
then presents the validation, the results, and the economics.

---

## 1. Problem statement

The model simulates a waterflood: water is injected into an oil-filled reservoir to
push oil toward a producing well. The simulator tracks two fields on a 2D grid
through time, the pressure $p$ and the water saturation $S_w$, and reports how much
oil is recovered and when water breaks through at the producer.

The test case is a quarter five-spot: an injector at one corner and a producer at the
opposite corner, with no-flow outer boundaries.

## 2. Assumptions

The governing equations below follow from a deliberate set of simplifications. Each is
standard for a first-principles waterflood model and each is defensible.

- Two immiscible phases, water and oil, no mass transfer between them.
- Incompressible rock and fluids (constant porosity, constant densities).
- Horizontal, areal (2D) flow, so gravity is neglected.
- Zero capillary pressure, so both phases share a single pressure field $p$.
- Isothermal.
- Absolute permeability may be homogeneous or spatially variable; porosity is constant.

The zero-capillary and no-gravity assumptions are the two most significant. They keep
the pressure equation to a single unknown pressure per cell and are appropriate for a
teaching-scale areal waterflood. Both can be added later without changing the overall
structure.

## 3. Governing equations

### 3.1 Darcy's law (per phase)

Each phase moves down the pressure gradient at a rate set by absolute permeability
$k$, that phase's relative permeability $k_{r\alpha}$, and its viscosity $\mu_\alpha$:

$$\mathbf{u}_\alpha = -\frac{k\,k_{r\alpha}}{\mu_\alpha}\,\nabla p = -\lambda_\alpha\, k\,\nabla p, \qquad \alpha = w, o$$

where $\lambda_\alpha = k_{r\alpha}/\mu_\alpha$ is the phase mobility.

### 3.2 Mass conservation (per phase)

For an incompressible phase, accumulation in the pore space balances the net flux out
plus any well source or sink:

$$\phi\,\frac{\partial S_\alpha}{\partial t} + \nabla \cdot \mathbf{u}_\alpha = q_\alpha$$

with the saturation constraint $S_w + S_o = 1$.

### 3.3 Splitting into a pressure equation and a saturation equation

Adding the water and oil conservation equations and using $S_w + S_o = 1$ (so the
time-derivative terms cancel for incompressible flow) removes saturation from the sum
and yields the **pressure equation**:

$$-\nabla \cdot \left(\lambda_t\, k\, \nabla p\right) = q_t, \qquad \lambda_t = \lambda_w + \lambda_o$$

This is elliptic. It has no time derivative, which is the mathematical statement that
in an incompressible medium a pressure change is felt everywhere at once. This is why
pressure is solved implicitly across the whole grid.

The water equation, rewritten with the total velocity
$\mathbf{u}_t = -\lambda_t k \nabla p$ and the fractional flow $f_w$, is the
**saturation (transport) equation**:

$$\phi\,\frac{\partial S_w}{\partial t} + \nabla \cdot \left(f_w\, \mathbf{u}_t\right) = q_w, \qquad f_w = \frac{\lambda_w}{\lambda_w + \lambda_o}$$

This is hyperbolic. It describes a saturation front advancing at a finite speed, which
is why saturation can be updated explicitly and locally.

The split of one elliptic pressure equation and one hyperbolic saturation equation is
the entire basis of the IMPES method.

### 3.4 Relative permeability (Corey model)

Relative permeabilities depend on a normalized water saturation that maps the mobile
range $[S_{wc},\, 1 - S_{or}]$ onto $[0, 1]$:

$$S_{wn} = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{or}}$$

$$k_{rw} = k_{rw}^{\max}\, S_{wn}^{\,n_w}, \qquad k_{ro} = k_{ro}^{\max}\, (1 - S_{wn})^{\,n_o}$$

As water saturation rises, $k_{rw}$ increases (water flows more easily) and $k_{ro}$
decreases (oil is choked off).

### 3.5 Mobility ratio

The end-point mobility ratio characterizes the displacement:

$$M = \frac{\lambda_w^{\,\max}}{\lambda_o^{\,\max}} = \frac{k_{rw}^{\max}/\mu_w}{k_{ro}^{\max}/\mu_o}$$

$M > 1$ is an unfavorable displacement: the injected water is more mobile than the oil
it displaces, so it fingers ahead, breaks through early, and leaves oil behind (poor
sweep efficiency). $M < 1$ is favorable and sweeps more like a piston.

## 4. Discretization

A cell-centered finite-volume scheme on a uniform $n_x \times n_y$ grid. Each cell has
bulk volume $V = \Delta x\, \Delta y\, h$.

### 4.1 Transmissibility

The transmissibility of a face measures how easily fluid crosses between two adjacent
cells. The geometric (rock-and-geometry) part in the x-direction is:

$$T^{\text{geo}}_{i+\frac12,j} = \beta\, k_{i+\frac12,j}\, \frac{\Delta y\, h}{\Delta x}$$

with the field-units constant $\beta = 1.127 \times 10^{-3}$. The face permeability uses
the **harmonic mean** of the two neighboring cell permeabilities:

$$k_{i+\frac12,j} = \frac{2\, k_{i,j}\, k_{i+1,j}}{k_{i,j} + k_{i+1,j}}$$

The harmonic mean is used because flow across the face passes through both cells in
series, and series flow is limited by the tighter (lower-permeability) rock. An
arithmetic mean would overstate the effective permeability. The geometric
transmissibility depends only on rock and geometry, so it is computed once.

The full transmissibility used in the flow equations multiplies in the total mobility
at the face, evaluated as the arithmetic average of the two cells:

$$T_{i+\frac12,j} = T^{\text{geo}}_{i+\frac12,j}\; \lambda_{t}^{\,\text{face}}$$

Mobility depends on saturation, which changes every step, so this part is recomputed
each step. (Arithmetic averaging of the face mobility for the pressure equation is a
common simplification; upstream weighting is the more rigorous alternative.)

### 4.2 Pressure equation: implicit solve

Applying mass conservation to each cell (net flux to neighbors equals the well source)
gives one linear equation per cell:

$$\sum_{\text{faces}} T_f\,(p_{\text{nb}} - p_c) + q_c = 0$$

Assembled over all $N = n_x n_y$ cells this is the linear system

$$\mathbf{A}\,\mathbf{p} = \mathbf{b}$$

The matrix $\mathbf{A}$ has each cell's neighbor transmissibilities as positive
off-diagonal entries and the negative sum of those transmissibilities on the diagonal.
The system is solved directly (sparse solver) so every cell pressure is obtained
simultaneously, which is the implicit step.

### 4.3 Well and boundary conditions

- **Injector:** rate-controlled. A fixed water rate $Q_{\text{inj}}$ enters the
  right-hand side $\mathbf{b}$ at the injector cell.
- **Producer:** pressure-controlled (Dirichlet). The producer cell's equation is
  replaced by $p = p_{\text{ref}}$.
- **Outer boundaries:** no-flow (no transmissibility across the domain edge).

Fixing the producer pressure is not optional. With both wells on rate control the
system would define only pressure differences, not absolute levels, leaving the
solution non-unique (a singular matrix). One fixed pressure anchors the field and makes
the solution unique.

### 4.4 Saturation equation: explicit update with upstream weighting

With pressure known, the signed total flux across each face follows from Darcy's law:

$$q_f = T_f\,(p_c - p_{\text{nb}})$$

The water flux across a face uses the fractional flow evaluated at the **upstream**
cell, the cell the flow is coming from:

$$q_{w,f} = f_w(S_{\text{up}})\; q_f$$

Upstream weighting is used because the fluid crossing a face physically carries the
upstream cell's composition. Using a downstream or averaged value numerically smears
the saturation front; upstream weighting keeps it sharp.

Summing the net water flux over each cell's faces (plus well terms) and stepping
forward in time gives the explicit update:

$$S_w^{\,n+1} = S_w^{\,n} + \frac{\Delta t \cdot 5.615}{\phi\, V}\; \left(\text{net water rate into cell, bbl/day}\right)$$

where $5.615$ converts barrels to cubic feet.

### 4.5 Stability: the CFL condition

Because the update is explicit and local, information propagates only one cell per time
step. The saturation front must therefore not travel more than about one cell per step,
which sets a maximum stable time step:

$$\Delta t \le \text{CFL} \cdot \frac{\phi\, V}{5.615 \cdot q_{\max} \cdot \max\left|\dfrac{d f_w}{d S_w}\right|}$$

The CFL safety factor (0.5 here) keeps the step comfortably below the limit. Exceeding
this limit makes the front outrun the scheme's one-cell reach and the solution goes
unstable (nonphysical oscillations, saturations outside physical bounds).

## 5. Reported quantities

- **Pore volumes injected:** $\text{PVI} = \dfrac{V_{\text{inj,cum}} \cdot 5.615}{V_{p,\text{total}}}$, the natural time axis for a waterflood.
- **Water cut** at the producer: $f_w$ evaluated at the producer cell, the fraction of produced fluid that is water.
- **Movable oil in place:** $\text{OOIP} = \dfrac{(1 - S_{wc} - S_{or})\, V_{p,\text{total}}}{5.615}$ (bbl).
- **Recovery factor:** cumulative produced oil divided by movable OOIP.

## 6. Solution algorithm (IMPES loop)

1. Initialize $S_w = S_{wc}$ everywhere (reservoir full of oil).
2. Solve the pressure equation implicitly at the current saturation.
3. Compute face fluxes from the pressure field.
4. Choose a stable time step from the CFL condition.
5. Update water saturation explicitly with upstream fractional flow.
6. Record water cut, recovery, and snapshots.
7. Repeat from step 2 until the target pore volumes have been injected.

## 7. Validation against the Buckley-Leverett analytical solution

A simulator is only worth trusting if it can reproduce a known answer. The 1D
Buckley-Leverett problem (incompressible, immiscible, zero capillary pressure)
has an exact analytical solution, so the simulator is run on a 1D grid with the
base-case fluid properties and compared against it.

The analytical solution comes from the Welge tangent construction. The shock
(front) saturation $S_{wf}$ is where the tangent from $(S_{wc}, 0)$ touches the
fractional flow curve, satisfying $f_w'(S_{wf}) = f_w(S_{wf})/(S_{wf}-S_{wc})$.
Behind the front is a rarefaction: each saturation $S \ge S_{wf}$ travels at
its own characteristic speed, $x_D = t_D\, f_w'(S)$, with $t_D$ measured in
pore volumes injected. Breakthrough occurs at $\text{PVI}_{bt} = 1/f_w'(S_{wf})$,
and the recovery at breakthrough follows from the Welge average saturation
$\bar{S}_w = S_{wf} + (1-f_w(S_{wf}))/f_w'(S_{wf})$.

Results for the base fluids ($M = 1.56$, 200 cells):

| Quantity | Analytical | Numerical | Error |
|---|---|---|---|
| Shock saturation $S_{wf}$ | 0.575 | matches profile | - |
| Breakthrough (PVI) | 0.462 | 0.456 | -1.2% |
| Recovery factor at breakthrough | 0.770 | 0.760 | -1.2% |

The numerical saturation profiles sit on top of the analytical rarefaction and
shock at both 0.15 and 0.30 PVI. The only visible difference is smearing of the
shock over a few cells, the expected numerical diffusion of first-order upwind
transport. A grid refinement study (50 to 400 cells) shows the front sharpening
toward the analytical discontinuity as $\Delta x$ shrinks, confirming the
scheme converges to the correct solution.

Every run also reports a discrete water material balance,

$$\varepsilon_{MB} = \frac{W_{\text{inj}} - W_{\text{prod}} - \Delta W_{\text{stored}}}{W_{\text{inj}}}$$

which comes back at machine precision ($\sim 10^{-14}$) in all cases. Mass is
conserved by construction, not by luck.

## 8. Results

### 8.1 Base case

The homogeneous quarter five-spot ($M = 1.56$) breaks through at 0.37 PVI and
recovers 84% of movable oil by 1 PVI. The saturation snapshots show the
characteristic five-spot front: radial near the injector, then cusping toward
the producer corner as the pressure field focuses the flow.

### 8.2 Mobility ratio

Sweeping oil viscosity from 0.5 to 10 cp spans end-point mobility ratios from
0.19 (favorable) to 3.89 (strongly unfavorable):

| $\mu_o$ (cp) | $M$ | Breakthrough (PVI) | RF at 1 PVI |
|---|---|---|---|
| 0.5 | 0.19 | 0.56 | 0.97 |
| 2.0 | 0.78 | 0.44 | 0.90 |
| 5.0 | 1.94 | 0.35 | 0.81 |
| 10.0 | 3.89 | 0.28 | 0.74 |

Rising $M$ pulls breakthrough earlier and cuts ultimate recovery, exactly the
piston-to-fingering transition the fractional flow theory predicts.

### 8.3 Heterogeneity

Three permeability fields with the same 200 mD average, same fluids ($M = 1.56$):

| Field | Breakthrough (PVI) | RF at 1 PVI |
|---|---|---|
| Homogeneous | 0.37 | 0.84 |
| High-perm channel (injector to producer) | 0.10 | 0.51 |
| Correlated log-normal ($V_{DP} = 0.7$) | 0.25 | 0.79 |

The channel case is the striking one: with identical average permeability,
water short-circuits down the channel, breaks through at one tenth of a pore
volume, and strands the off-channel oil, cutting recovery by a third. The
lesson is that the average permeability of an asset says almost nothing about
its flood performance; contrast and connectivity control sweep. This is why
reserves attributed to a waterflood depend on geology, not just volumetrics.

## 9. Layer 2: asset economics and NPV optimization

The production stream from the simulator feeds a monthly discounted cash flow
model: flat price deck, net revenue interest, fixed and variable opex, water
handling cost on both injected and produced water, and capex split into
drilling plus rate-dependent facilities. Production truncates at the economic
limit, the first month operating cash flow goes negative, so estimated ultimate
recovery (EUR) is an output of the economics rather than the physics alone.

A useful property of incompressible flow makes rate optimization essentially
free: the saturation solution depends only on pore volumes injected, so one
simulation provides the water cut and recovery curves for every injection rate,
each with its own time axis $t = \text{PVI} \cdot V_p / (5.615\, Q)$.

Sweeping the injection rate exposes the central trade-off. Higher rates pull
the same barrels forward in time (less discounting) but require larger
facilities capex, while low rates burn fixed opex for years. With the base
assumptions (70 $/bbl, 25% royalty, 10% discount rate) NPV peaks at an interior
optimum of 750 bbl/day, while EUR keeps creeping upward all the way to the
highest rate tested. Designing the flood for maximum barrels instead of maximum
value would surrender about $2.2MM of NPV on a $5.6MM asset. More oil is not
more value, and the gap between the two is where acquisition and divestiture
judgment lives.

## 10. Limitations and extensions

The model is deliberately minimal, and the honest list of what it leaves out is
part of understanding it:

- **No compressibility.** Real reservoirs store energy in fluid and rock
  compression; this model has no primary depletion, only displacement. Adding
  compressibility would also break the rate-scaling shortcut used in the
  economics layer.
- **No gravity or capillary pressure.** Fine for a thin areal flood; wrong for
  thick reservoirs or transition zones. Capillary pressure would also
  regularize the saturation shock physically rather than numerically.
- **Simple well model.** The producer is a Dirichlet cell rather than a
  Peaceman well index, so near-well pressure is grid-dependent. A Peaceman
  model is the standard fix.
- **First-order transport.** Upwind differencing smears the front over a few
  cells. Higher-order TVD schemes would sharpen it at the same grid size.
- **IMPES stability.** The explicit saturation update carries a CFL limit; a
  fully implicit or adaptive-implicit formulation would allow larger steps.
- **Flat price deck and deterministic inputs.** The natural next layer is
  Monte Carlo over permeability, OOIP, and price to produce P10/P50/P90
  distributions of reserves and value, which is how assets are actually
  evaluated in A&D.

## 11. Nomenclature

| Symbol | Meaning | Units |
|---|---|---|
| $p$ | pressure | psi |
| $S_w, S_o$ | water, oil saturation | fraction |
| $S_{wc}, S_{or}$ | connate water, residual oil | fraction |
| $S_{wn}$ | normalized water saturation | fraction |
| $k$ | absolute permeability | mD |
| $k_{r\alpha}$ | relative permeability of phase $\alpha$ | fraction |
| $\mu_\alpha$ | viscosity of phase $\alpha$ | cp |
| $\lambda_\alpha$ | phase mobility, $k_{r\alpha}/\mu_\alpha$ | 1/cp |
| $\lambda_t$ | total mobility | 1/cp |
| $f_w$ | fractional flow of water | fraction |
| $M$ | end-point mobility ratio | dimensionless |
| $\phi$ | porosity | fraction |
| $T_f$ | face transmissibility | bbl/day/psi |
| $q$ | volumetric rate (well or face) | bbl/day |
| $\beta$ | Darcy constant (field units) | 1.127e-3 |
| $V$ | cell bulk volume | ft³ |
| $V_p$ | pore volume | ft³ |

---

*Implemented in Python (NumPy, SciPy sparse solver, Matplotlib, pandas). The pressure
solve uses a sparse direct solver; the transport update is fully vectorized. Code,
studies, and figures: see the repository README.*
