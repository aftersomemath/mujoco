# System Identification Toolbox

Given a MuJoCo model and recorded sensor data, find the physical parameters
that make simulation match reality. The library solves a box-constrained
nonlinear least-squares problem, minimizing the weighted residual
`½ ‖W(ȳ(θ) − y)‖²` between simulated output `ȳ(θ)` and recorded
measurements `y`.

The optimizer uses Gauss-Newton with finite-difference Jacobians. Each
parameter perturbation requires an independent simulation rollout, and all
of them execute in a single batched call to `mujoco.rollout`, parallelized
across threads.

## Pipeline

**You provide:**
- One or more `ModelSequences` bundling your `MjSpec` with measured data.
  Multiple `ModelSequences` with different specs can be optimized jointly.
  For example, recordings of the same robot with and without a known payload
  can be combined for a better-conditioned problem.
- A `ParameterDict` defining what to identify, with bounds.

**The framework:**
- Builds a residual function (`build_residual_fn`).
- Optimizes parameters via batched parallel rollouts (`optimize`).
- Saves results and generates an HTML report (`save_results`, `default_report`).

## What Can You Identify?

**Physics parameters** can be anything settable on `MjSpec`. The library
provides convenience functions for common system identification targets such
as body inertia and actuator gains. Anything else can be set directly on the
spec with a modifier callback:

| Target | Approach |
|---|---|
| Body mass | `body_inertia_param(..., InertiaType.Mass)` |
| Body mass + center of mass | `body_inertia_param(..., InertiaType.MassIpos)` |
| Full body inertia (10-D) | `body_inertia_param(..., InertiaType.Pseudo)` |
| Actuator P/D gains | `apply_pgain(spec, "act1", p.value[0])` |
| Contact sliding friction | `spec.pair("cp").friction[0] = p.value[0]` |
| Joint damping | `spec.joint("j1").damping = p.value[0]` |

Full inertia uses the pseudo-inertia Cholesky parameterization
([Rucker & Wensing 2022](https://ieeexplore.ieee.org/document/9690029)),
guaranteeing physical consistency for any `θ`.

**Measurement parameters** such as sensor delays, gains, and biases are
properties of the measurement system, not the physics model. The library
provides utilities for applying these corrections to the residual after
rollout.

## Example

```python
import mujoco
from mujoco import sysid

# 1. Load model and define parameters.
spec = mujoco.MjSpec.from_file("robot.xml")
model = spec.compile()

def set_link1_mass(spec, p):
    spec.body("link1").mass = p.value[0]

params = sysid.ParameterDict()
params.add(sysid.Parameter(
    "link1_mass", nominal=2.0, min_value=0.5, max_value=5.0,
    modifier=set_link1_mass))

# 2. Load and package measured data.
control = sysid.TimeSeries.from_control_names(times, ctrl_array, model)
sensordata = sysid.TimeSeries.from_names(times, sensor_array, model)
initial_state = sysid.create_initial_state(model, qpos_0, qvel_0)
ms = sysid.ModelSequences("robot", spec, "traj_1", initial_state, control, sensordata)

# 3. Build residual, optimize, save.
residual_fn = sysid.build_residual_fn(models_sequences=[ms])
opt_params, opt_result = sysid.optimize(initial_params=params, residual_fn=residual_fn)
sysid.save_results("results/", [ms], params, opt_params, opt_result, residual_fn)
```

`default_report` generates an interactive HTML report with sensor comparisons,
parameter tables, confidence intervals, and rollout videos.
