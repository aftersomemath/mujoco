# System Identification Toolbox

Identify physical parameters of a MuJoCo model from recorded sensor data.
Given parameters `θ`, the library minimizes `½ ‖W(ȳ(θ) − y)‖²` subject to
box constraints, where `ȳ` is simulated and `y` is recorded sensor output.

The optimizer uses Gauss-Newton with finite-difference Jacobians. Every
perturbed rollout needed for the Jacobian is independent, so they all execute
in a single batched call to `mujoco.rollout`, fully parallelized across
cores with no user effort.

## Pipeline

```
Define Parameters → Package Data → Build Residual → Optimize → Report
  ParameterDict     ModelSequences  build_residual_fn  optimize   default_report
```

## What Can You Identify?

**Physics parameters** can be anything settable on `MjSpec` via a modifier callback:

| Target | Approach |
|---|---|
| Body mass | `body_inertia_param(..., InertiaType.Mass)` |
| Body mass + center of mass | `body_inertia_param(..., InertiaType.MassIpos)` |
| Full body inertia (10-D) | `body_inertia_param(..., InertiaType.Pseudo)` |
| Actuator P/D gains | `apply_pgain(spec, "act1", p.value[0])` |
| Contact friction | `spec.pair("cp").friction[0] = p.value[0]` |
| Joint damping | `spec.joint("j1").damping = p.value[0]` |

Full inertia uses the pseudo-inertia Cholesky parameterization
([Rucker & Wensing 2022](https://ieeexplore.ieee.org/document/9690029)),
guaranteeing physical consistency for any `θ`.

**Measurement parameters** such as sensor delays, gains, and biases are
configured declaratively via `SignalTransform`:

| Target | Approach |
|---|---|
| Sensor delay | `transform.delay("*_pos", params["delay"])` |
| Sensor gain | `transform.gain("*_torque", params["scale"])` |
| Sensor bias | `transform.bias("*_vel", params["bias"])` |

Multiple `ModelSequences` with different specs can be optimized jointly.
For example, recordings of the same robot with and without a known payload
can share inertial parameters, stacking residuals for a better-conditioned
problem.

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
    modifier=set_link1_mass,
))

# 2. Package recorded data.
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
