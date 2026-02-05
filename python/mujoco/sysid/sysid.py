# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build and solve the system identification problem."""

import copy
import dataclasses
import os
import pathlib
import pickle
from typing import Any, Callable, List, Literal, Mapping, Optional, Sequence, TypeAlias, Union, Dict, Tuple

from absl import logging
import matplotlib.pyplot as plt
import mujoco
from mujoco import minimize as mujoco_minimize
import mujoco.rollout as mj_rollout
import numpy as np
import scipy.optimize as scipy_optimize
from scipy.special import stdtrit
import tqdm

from mujoco.sysid import parameter
from mujoco.sysid import plotting
from mujoco.sysid import model_modifier
from mujoco.sysid import signal_modifier
from mujoco.sysid import timeseries
from mujoco.sysid.report.builder import ReportBuilder

from mujoco.sysid.report.sections.parameters import ParametersTable
from mujoco.sysid.report.sections.covariance import Covariance
from mujoco.sysid.report.sections.signals import SignalReport
from mujoco.sysid.report.sections.optimization_trace import OptimizationTrace

BuildModelFn: TypeAlias = Callable[
        [parameter.ParameterDict, mujoco.MjSpec], mujoco.MjModel
    ]
CustomRolloutFn: TypeAlias = Any  # TODO(kevin): Fix this.
ModifyResidualFn: TypeAlias = Any  # TODO(kevin): Fix this.


@dataclasses.dataclass(frozen=True)
class SystemTrajectory:
  """Encapsulates a trajectory rolled out from a system.

  Attributes:
    model: MuJoCo model used to simulate the trajectory.
    control: A TimeSeries instance containing control signals.
    sensordata: A TimeSeries instance containing sensor data.
    initial_state: Initial state of the simulation. Shape (n_state,).
    state: Simulation states over time. Shape (n_steps, n_state). Optional for
      real robot trajectories.
  """

  model: mujoco.MjModel
  control: timeseries.TimeSeries
  sensordata: timeseries.TimeSeries
  initial_state: np.ndarray
  state: timeseries.TimeSeries | None

  def replace(self, **kwargs) -> "SystemTrajectory":
    return dataclasses.replace(self, **kwargs)

  def get_sensordata_slice(self, sensor: str = "joint_pos") -> np.ndarray:
    if sensor == "joint_pos":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTPOS
    elif sensor == "joint_vel":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTVEL
    elif sensor == "joint_torque":
      sensor_type = mujoco.mjtSensor.mjSENS_JOINTACTFRC
    else:
      raise ValueError(f"Unsupported sensor type: {sensor}")
    adr = []
    dims = []
    for i in range(self.model.nsensor):
      if self.model.sensor(i).type == sensor_type:
        sensor_id = self.model.sensor(i).id
        adr.append(self.model.sensor_adr[sensor_id])
        dims.append(self.model.sensor_dim[sensor_id])
    sensors = sorted(zip(adr, dims), key=lambda x: x[0])
    start = sensors[0][0]
    total_dim = sum(d for _, d in sensors)
    end = start + total_dim
    return self.sensordata.data[:, start:end]

  @property
  def sensordim(self) -> int:
    return self.model.nsensordata

  def __len__(self) -> int:
    return len(self.sensordata)

  def save_to_disk(self, path: pathlib.Path) -> None:
    save_dict = {
        "control_times": self.control.times,
        "control_data": self.control.data,
        "sensordata_times": self.sensordata.times,
        "sensordata_data": self.sensordata.data,
        "initial_state": self.initial_state,
    }
    if self.state is not None:
      save_dict["state_times"] = self.state.times
      save_dict["state_data"] = self.state.data
      save_dict["state_signal_mapping"] = np.array(self.state.signal_mapping, dtype=object)

    if self.control.signal_mapping:
      save_dict["control_signal_mapping"] = np.array(self.control.signal_mapping, dtype=object)

    if self.sensordata.signal_mapping:
      save_dict["sensordata_signal_mapping"] = np.array(self.sensordata.signal_mapping, dtype=object)

    np.savez(path, **save_dict)  # type: ignore

  @classmethod
  def load_from_disk(
      cls,
      path: pathlib.Path,
      model: mujoco.MjModel,
      allow_missing_sensors: bool = False,
  ) -> "SystemTrajectory":
    with np.load(path, allow_pickle=True) as npz:
      control_times = npz["control_times"]
      control_data = npz["control_data"]
      sensordata_times = npz["sensordata_times"]
      sensordata_data = npz["sensordata_data"]
      initial_state = npz["initial_state"]
      state_times = npz.get("state_times", None)
      state_data = npz.get("state_data", None)

      control_signal_mapping = None
      if "control_signal_mapping" in npz:
        control_signal_mapping = npz["control_signal_mapping"].item()

      sensordata_signal_mapping = None
      if "sensordata_signal_mapping" in npz:
        sensordata_signal_mapping = npz["sensordata_signal_mapping"].item()

      state_signal_mapping = None
      if "state_signal_mapping" in npz:
        state_signal_mapping = npz["state_signal_mapping"].item()

    predicted_rollout = cls(
        model=model,
        control=timeseries.TimeSeries(control_times, control_data, signal_mapping=control_signal_mapping),
        sensordata=timeseries.TimeSeries(sensordata_times, sensordata_data, signal_mapping=sensordata_signal_mapping),
        initial_state=initial_state,
        state=timeseries.TimeSeries(state_times, state_data, signal_mapping=state_signal_mapping) if state_times is not None else None,
    )
    predicted_rollout.check_compatible(allow_missing_sensors)
    return predicted_rollout

  def check_compatible(self, allow_missing_sensors: bool = False):
    if self.sensordata.data.shape[1] != self.model.nsensordata:
      if not allow_missing_sensors:
        raise ValueError(
            f"Sensor data dimension {self.sensordata.data.shape[1]} does not"
            f" match model sensor dimension {self.model.nsensordata}"
        )
      else:
        logging.warning(
            f"Sensor data dimension {self.sensordata.data.shape[1]} does not"
            f" match model sensor dimension {self.model.nsensordata}"
        )

    if self.control.data.shape[1] != self.model.nu:
      raise ValueError(
          f"Control data dimension {self.controldata.data.shape[1]} does not"
          f" match model control dimension {self.model.nu}"
      )

    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(self.model, state_spec.value)
    if self.state is not None:
      if self.state.data.shape[1] != state_size:
        raise ValueError(
            f"State dimension {self.state.data.shape[1]} does not match "
            f"model state dimension {state_size}"
        )
    if self.initial_state.shape[0] != state_size:
      raise ValueError(
          f"Initial state dimension {self.initial_state.shape[0]} does not"
          f" match model state dimension {state_size}"
      )

  def split(self, chunk_size: int) -> list["SystemTrajectory"]:
    if self.state is None:
      raise ValueError("Cannot split rollout with missing state field.")
    steps = len(self.sensordata.times)
    n_complete_chunks = steps // chunk_size
    control_times = self.control.times
    control_data = self.control.data
    sensordata_times = self.sensordata.times
    sensordata_data = self.sensordata.data
    trajectories = []
    for i in range(n_complete_chunks):
      start_idx = i * chunk_size
      end_idx = start_idx + chunk_size
      initial_state = (
          self.initial_state if start_idx == 0 else self.state.data[start_idx - 1]
      )
      control_times_chunk = control_times[start_idx:end_idx]
      control_data_chunk = control_data[start_idx:end_idx]
      sensordata_times_chunk = sensordata_times[start_idx:end_idx]
      sensordata_data_chunk = sensordata_data[start_idx:end_idx]
      trajectories.append(
          SystemTrajectory(
              model=self.model,
              control=timeseries.TimeSeries(
                  control_times_chunk, control_data_chunk
              ),
              sensordata=timeseries.TimeSeries(
                  sensordata_times_chunk, sensordata_data_chunk
              ),
              initial_state=initial_state,
              state=self.state.data[start_idx:end_idx],
          )
      )
    return trajectories

  def render(
      self,
      height: int = 240,
      width: int = 320,
      camera: str | int = -1,
      fps: int = 30,
  ) -> List[np.ndarray]:
    if self.state is None:
      raise ValueError("Cannot render rollout with missing state field.")

    # Adapt state to batch format (nbatch=1, nsteps, nstate)
    state_batch = self.state.data[np.newaxis, :, :]

    data = mujoco.MjData(self.model)

    return render_rollout(
        model=self.model,
        data=data,
        state=state_batch,
        framerate=fps,
        camera=camera,
        width=width,
        height=height
    )


def render_rollout(
    model: mujoco.MjModel | Sequence[mujoco.MjModel],
    data: mujoco.MjData,
    state: np.ndarray,
    framerate: int,
    camera: str | int = -1,
    width: int = 640,
    height: int = 480,
    light_pos: Optional[Sequence[float]] = None,
) -> List[np.ndarray]:
  """Renders a rollout or batch of rollouts.

  Args:
    model: Single model or list of models (one per batch).
    data: MjData scratch object.
    state: State array of shape (nbatch, nsteps, nstate).
    framerate: Frames per second to render.
    camera: Camera name or ID.
    width: Image width.
    height: Image height.
    light_pos: Optional light position [x, y, z] to add a spotlight.

  Returns:
    List of rendered frames (numpy arrays).
  """
  nbatch = state.shape[0]

  if not isinstance(model, mujoco.MjModel):
    model = list(model)

  if isinstance(model, list) and len(model) == 1:
    model = model * nbatch
  elif isinstance(model, list):
    assert len(model) == nbatch
  else:
    model = [model] * nbatch

  # Visual options
  vopt = mujoco.MjvOption()
  vopt.geomgroup[3] = 1 # Show visualization geoms

  pert = mujoco.MjvPerturb()
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  # Simulate and render.
  frames = []

  with mujoco.Renderer(model[0], height=height, width=width) as renderer:
    for i in range(state.shape[1]):
      # Check if we should capture this frame based on framerate
      if len(frames) < i * model[0].opt.timestep * framerate:
        for j in range(state.shape[0]):
          # Set state
          mujoco.mj_setState(model[j], data, state[j, i, :],
                             mujoco.mjtState.mjSTATE_FULLPHYSICS)
          mujoco.mj_forward(model[j], data)

          # Use first model to make the scene, add subsequent models
          if j == 0:
            renderer.update_scene(data, camera, scene_option=vopt)
          else:
            mujoco.mjv_addGeoms(model[j], data, vopt, pert, catmask, renderer.scene)

        # Add light, if requested
        if light_pos is not None:
          if renderer.scene.nlight < 100: # check limit
            light = renderer.scene.lights[renderer.scene.nlight]
            light.ambient = [0, 0, 0]
            light.attenuation = [1, 0, 0]
            light.castshadow = 1
            light.cutoff = 45
            light.diffuse = [0.8, 0.8, 0.8]
            light.dir = [0, 0, -1]
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            light.exponent = 10
            light.headlight = 0
            light.specular = [0.3, 0.3, 0.3]
            light.pos = light_pos
            renderer.scene.nlight += 1

        # Render and add the frame.
        pixels = renderer.render()
        frames.append(pixels)
  return frames


def create_initial_state(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: Optional[np.ndarray] = None,
    act: Optional[np.ndarray] = None,
) -> np.ndarray:
  data = mujoco.MjData(model)
  initial_state = np.empty((
      mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS),
  ))
  if qpos.shape[0] != model.nq:
    raise ValueError(
        f"Expected qpos to have shape {model.nq}, got {qpos.shape[0]}."
    )
  data.qpos[:] = qpos
  if qvel is not None:
    if qvel.shape[0] != model.nv:
      raise ValueError(
          f"Expected qvel to have shape {model.nv}, got {qvel.shape[0]}."
      )
    data.qvel[:] = qvel
  if act is not None:
    if act.shape[0] != model.na:
      raise ValueError(
          f"Expected act to have shape {model.na}, got {act.shape[0]}."
      )
    data.act[:] = act
  mujoco.mj_getState(
      model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
  )
  return initial_state


class ModelSequences:

  def __init__(
      self,
      name: str,
      spec: mujoco.MjSpec,
      sequence_name: Union[str, Sequence[str]],
      initial_state: Union[np.ndarray, Sequence[np.ndarray]],
      control: Union[timeseries.TimeSeries, Sequence[timeseries.TimeSeries]],
      sensordata: Union[timeseries.TimeSeries, Sequence[timeseries.TimeSeries]],
      allow_missing_sensors: bool = False,
  ):
    self.name = name
    self.spec = spec
    self.sequence_name = sequence_name
    self.initial_state = initial_state
    self.control = control
    self.sensordata = sensordata
    self.allow_missing_sensors = allow_missing_sensors

    self.gt_model = self.spec.compile()

    if isinstance(self.sequence_name, str):
      self.sequence_name = [self.sequence_name]
    if isinstance(self.initial_state, np.ndarray):
      self.initial_state = [self.initial_state]
    if isinstance(self.control, timeseries.TimeSeries):
      self.control = [self.control]
    if isinstance(self.sensordata, timeseries.TimeSeries):
      self.sensordata = [self.sensordata]

    self.measured_rollout = []
    for initial_state_, control_, sensordata_ in zip(
        self.initial_state, self.control, self.sensordata
    ):
      measured_rollout_ = SystemTrajectory(
          model=self.gt_model,
          control=control_,
          sensordata=sensordata_,
          initial_state=initial_state_,
          state=None,
      )
      measured_rollout_.check_compatible(allow_missing_sensors=allow_missing_sensors)
      self.measured_rollout.append(measured_rollout_)

  def __getitem__(self, key):
    return ModelSequences(
        self.name,
        self.spec,
        self.sequence_name[key],
        self.initial_state[key],
        self.control[key],
        self.sensordata[key],
        self.allow_missing_sensors,
    )


def _timeseries2array(
    control_signal: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
) -> tuple[np.ndarray, np.ndarray]:
  if isinstance(control_signal, timeseries.TimeSeries):
    control = control_signal.data
    control_times = control_signal.times
  else:
    control = np.stack([ts.data for ts in control_signal], axis=0)
    control_times = np.stack([ts.times for ts in control_signal], axis=0)
  # The measured data has N sensor measurements and N controls, where the first sensor
  # measurement corresponds to the initial condition. Thus we don't have ground truth
  # for the N+1'th state produced by the N'th control and so there is no point in
  # simulating it.
  if control.ndim == 3:
    control_applied_times = control_times[:, :-1]
    control_applied = control[:, :-1, :]
  else:
    control_applied_times = control_times[:-1]
    control_applied = control[:-1, :]
  return control_applied, control_applied_times


def _sequence2array(
    initial_states: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
  if isinstance(initial_states, np.ndarray):
    return initial_states
  return np.stack(initial_states, axis=0)


def _arrays2traj(
    models: mujoco.MjModel | Sequence[mujoco.MjModel],
    initial_states: np.ndarray | Sequence[np.ndarray],
    control: np.ndarray,
    control_times: np.ndarray,
    state: np.ndarray,
    sensordata: np.ndarray,
    signal_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]],
    state_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]],
    ctrl_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]],
) -> Sequence[SystemTrajectory]:
  # Add a batch dimension, no-op if already present.
  nbatch = state.shape[0]
  initial_states = np.tile(initial_states, (nbatch, 1))
  control = np.tile(control, (nbatch, 1, 1))
  control_times = np.tile(control_times, (nbatch, 1))

  obs_model = None
  if isinstance(models, mujoco.MjModel):
    models = [models] * nbatch
    obs_model = models
  else:
    obs_model = models[0]

  return [
      SystemTrajectory(
          model=models[i],
          control=timeseries.TimeSeries(control_times[i], control[i], signal_mapping=ctrl_mapping),
          # NOTE(kevin): When using mjSTATE_FULLPHYSICS, the first element of
          # the state corresponds to the simulation time. The reason we do not
          # use control_times[i] is because sensordata times are shifted by
          # one time step.
          sensordata=timeseries.TimeSeries(state[i][:, 0], sensordata[i], signal_mapping) if sensordata[i].size > 0 else None,
          initial_state=initial_states[i],
          state = timeseries.TimeSeries(times=state[i][:, 0], data=state[i], signal_mapping=state_mapping)
      )
      for i in range(nbatch)
  ]


def construct_ts_from_defaults(state_ts: timeseries.TimeSeries,
                               pred_sensordata: timeseries.TimeSeries,
                               measured_sensordata: timeseries.TimeSeries,
                               enabled_observations: List[Tuple[str, timeseries.SignalType]] | None = None):

  # Trim measured data enabled observations
  if enabled_observations:
    enabled_observations_names = [i[0] for i in enabled_observations]
    enabled_observations_types = [i[1] for i in enabled_observations]
  else:
    enabled_observations_names = measured_sensordata.signal_mapping.keys()
    enabled_observations_types = list(measured_sensordata.signal_mapping.values())
    enabled_observations_types = [i[0] for i in enabled_observations_types]


  selected_measured_sensordata = timeseries.TimeSeries.slice_by_name(measured_sensordata, enabled_observations_names)

  shape = (pred_sensordata.data.shape[0], selected_measured_sensordata.data.shape[1])
  modified_predictedulated_data = np.zeros(shape)

  measured_signal_mapping = measured_sensordata.signal_mapping
  for enabled_obs_name, enabled_obs_type in zip(enabled_observations_names, enabled_observations_types):

    if enabled_obs_name not in measured_signal_mapping and enabled_obs_name not in state_ts.signal_mapping:
      raise ValueError(f"{enabled_obs_name} is missing.")

    obs_type, indices = measured_signal_mapping[enabled_obs_name]

    if obs_type != enabled_obs_type:
      raise ValueError(f"Observation type error: {enabled_obs_name} is of type {obs_type} but declared as {enabled_obs_type}.")

    if obs_type == timeseries.SignalType.CustomObs:
      raise ValueError(f"You are attempting to use the default SysID's modify_residual with a custom observation of name {enabled_obs_name}. This is not supported. You must implement your own modify_residual. See documentation at ...")

    elif obs_type == timeseries.SignalType.MjSensor:
      target_indices = selected_measured_sensordata.signal_mapping[enabled_obs_name][1]
      modified_predictedulated_data[:, ..., target_indices] = pred_sensordata.data[:,...,indices]

    elif obs_type == timeseries.SignalType.MjStateQPos or obs_type == timeseries.SignalType.MjStateQVel or timeseries.SignalType.MjStateAct:
      state_indices = state_ts.signal_mapping[enabled_obs_name][1]

      values = state_ts.data[:, ..., state_indices]
      target_indices = selected_measured_sensordata.signal_mapping[enabled_obs_name][1]
      modified_predictedulated_data[:, ..., target_indices] = values


  ts_predicted_data = timeseries.TimeSeries(pred_sensordata.times, modified_predictedulated_data, selected_measured_sensordata.signal_mapping)

  return selected_measured_sensordata, ts_predicted_data


def sysid_rollout(
    models: mujoco.MjModel | Sequence[mujoco.MjModel],
    datas: mujoco.MjData | Sequence[mujoco.MjData],
    control_signal: Sequence[timeseries.TimeSeries] | timeseries.TimeSeries,
    initial_states: np.ndarray | Sequence[np.ndarray],
    rollout_signal_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]] | None = None,
    rollout_state_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]] | None = None,
    ctrl_mapping: Dict[str, Tuple[timeseries.SignalType, np.ndarray]] | None = None,
) -> Sequence[SystemTrajectory]:
  """Rollout trajectories in parallel for the given models and controls.

  Args:
    models: MuJoCo model or sequence of models.
    datas: MuJoCo data or sequence of data.
    control_signal: Control signals as TimeSeries or sequence of TimeSeries.
    initial_states: Initial states of the simulation. Shape (n_state,) or
      (n_batch, n_state).
    state_indices: Indices of states that should be treated as sensor data.

  Returns:
    Sequence of SystemTrajectory instances containing the simulation results.
  """

  # if the user does not supply it, we create it. Note that this will impact perf.
  if not rollout_signal_mapping or not rollout_state_mapping or not ctrl_mapping:
    model0 = models
    if not isinstance(models, mujoco.MjModel):
      model0 = models[0]
    qpos_map, qvel_map, act_map, ctrl_mapping = timeseries.get_state_maps(model0)
    rollout_state_mapping = qpos_map | qvel_map | act_map
    rollout_signal_mapping = timeseries.mj_sensor_to_map(model0)

  control, control_times = _timeseries2array(control_signal)
  initial_states = _sequence2array(initial_states)
  state, sensordata = mj_rollout.rollout(models, datas, initial_states, control)

  return _arrays2traj(
      models, initial_states, control, control_times, state, sensordata, rollout_signal_mapping, rollout_state_mapping, ctrl_mapping
  )


# Lowest level residual function, works on one model
def model_residual(
    x: np.ndarray,
    params: parameter.ParameterDict,
    build_model: BuildModelFn,
    traj_measured: Sequence[SystemTrajectory] | SystemTrajectory,
    modify_residual: Optional[ModifyResidualFn] = None,
    custom_rollout: Optional[CustomRolloutFn] = None,
    n_threads: int = os.cpu_count(),
    return_pred_all: bool = False,
    resample_true: bool = True,
    sensor_weights: Optional[Mapping[str, float]] = None,
    enabled_observations: List[Tuple[str, timeseries.SignalType]] = []
):
  """Compute a vector of residuals for a given parameter vector.

  Args:
    x: Current decision variable vector.
    params: Parameter dictionary.
    build_model: Function to build a MuJoCo model from parameters.
    n_threads: Number of threads for parallel computation.
    traj_measured: Ground truth trajectory or sequence of trajectories.
    modify_residual: Function to modify the residuals of a predicted and measured
      trajectory.
    custom_rollout: Function to override the built in rollout implementation
    return_pred_all: Return the final predicted and measured sensor data for all
      trajectories.
    resample_measured: If modify_residual not passed, whether to resample measured data
      at sim times.
    sensor_weights: An optional dict mapping sensor name to weight. Unspecified sensors
      are assumed to have a weight of 1.
    enabled_observations: An optional list of tuples containing observation and their obs-
      ervation type.

  Returns:
    Vector of residuals if return_pred is False, otherwise a tuple of the residuals,
    the predicted sensor data, and the ground truth sensor data.
  """
  # Convert single trajectory to list for consistent handling.
  if isinstance(traj_measured, SystemTrajectory):
    traj_measured = [traj_measured]
  n_chunks = len(traj_measured)

  # Handle finite difference columns if present.
  initial_ndim = x.ndim
  n_fd = 1
  if x.ndim > 1:
    n_fd = x.shape[1]
    x_reshaped = x
  else:
    x_reshaped = x.reshape(-1, 1)

  # Process each finite difference column.
  models = []
  models_x = []
  model_0 = None
  for i in range(n_fd):
    params.update_from_vector(x_reshaped[:, i])
    model = build_model(params)
    if not model_0:
      model_0 = model
    models_x.extend([x_reshaped[:, i]] * n_chunks)
    models.extend([model] * n_chunks)

  qpos_map, qvel_map, act_map, rollout_ctrl_map = timeseries.get_state_maps(model_0)
  rollout_state_mapping = qpos_map | qvel_map | act_map
  rollout_signal_mapping = timeseries.mj_sensor_to_map(model_0)

  # Create data objects for parallel computation.
  datas = [mujoco.MjData(models[0]) for _ in range(n_threads)]

  # Interpolate control signal.
  if resample_true:
    control_chunks = [
        traj.control.resample(target_dt=models[0].opt.timestep)
        for traj in traj_measured
    ]
  else:
    control_chunks = [traj.control for traj in traj_measured]

  # Rollout trajectories in parallel.
  if custom_rollout is None:
    pred_trajectories = sysid_rollout(
        models=models[: n_fd * n_chunks],
        datas=datas,
        control_signal=[control for control in control_chunks] * n_fd,
        initial_states=[chunk.initial_state for chunk in traj_measured] * n_fd,
        rollout_signal_mapping=rollout_signal_mapping,
        rollout_state_mapping=rollout_state_mapping,
        ctrl_mapping=rollout_ctrl_map
    )
  else:
    param_dicts = [copy.deepcopy(params) for i in range(x_reshaped.shape[1])]
    [
        param_dicts[i].update_from_vector(x_reshaped[:, i])
        for i in range(x_reshaped.shape[1])
    ]
    pred_trajectories = custom_rollout(
        models=models[: n_fd * n_chunks],
        datas=datas,
        control_signal=[control for control in control_chunks] * n_fd,
        initial_states=[chunk.initial_state for chunk in traj_measured] * n_fd,
        param_dicts=param_dicts,
        rollout_signal_mapping=rollout_signal_mapping,
        rollout_state_mapping=rollout_state_mapping,
        ctrl_mapping=rollout_ctrl_map
    )

  # Compute residuals for each trajectory chunk.
  all_residuals = []
  pred_sensordatas = []
  measured_sensordatas = []

  for i in range(len(models)):
    model = models[i]
    pred_traj = pred_trajectories[i]
    pred_state = pred_trajectories[i].state.data

    rollout_state_ts = timeseries.TimeSeries(times=pred_state[:, 0], data=pred_state[:, 1:], signal_mapping=rollout_state_mapping)

    measuredidx = i % n_chunks
    measuredtraj = traj_measured[measuredidx]

    pred_sensordata = pred_traj.sensordata
    measured_sensordata = measuredtraj.sensordata

    # If the user passes a residual function allow them to handle all resampling, etc.
    if modify_residual is not None:

      params.update_from_vector(models_x[i])
      res, pred_sensordata, measured_sensordata = modify_residual(
          params, pred_sensordata, measured_sensordata, model, return_pred_all, state=pred_state
      )

    # If the user does not pass a residual function, resample the ground truth data to
    # match the sime times if requested.
    else:

      measured_sensordata, pred_sensordata = construct_ts_from_defaults(rollout_state_ts, pred_sensordata, measured_sensordata, enabled_observations)
      if resample_true:
        # Window the true data so that times in it correspond to times spanned by
        # predicted data.
        measured_sensordata = signal_modifier.apply_delayed_ts_window(
            measured_sensordata, pred_sensordata, 0.0, 0.0
        )
        # Sample the predicted signal at the true times.
        pred_sensordata = pred_sensordata.resample(measured_sensordata.times)

      else:
        # Do not include difference in first sensor outputs in residual vector.
        # It corresponds to the initial condition and so provides little new
        # information. Additionally the semantics of rollout make it difficult to
        # simulate the sensor output corresponding to the initial condition.
        measured_sensordata = timeseries.TimeSeries(
            measured_sensordata.times[1:], measured_sensordata.data[1:, :], measured_sensordata.signal_mapping
        )

      res = signal_modifier.weighted_diff(
          predicted_data=pred_sensordata.data,
          measured_data=measured_sensordata.data,
          model=model,
          sensor_weights=sensor_weights,
      )
      res = signal_modifier.normalize_residual(res, measured_sensordata.data)

    if pred_sensordata.signal_mapping != measured_sensordata.signal_mapping:
      raise ValueError(f"The observation mapping between the measured data and predicted rollout data is not the same. You have not modified the observation data in TimeSeries in modify_residual to correctly reflect the measured data.")

    all_residuals.append(res)
    pred_sensordatas.append(pred_sensordata)
    measured_sensordatas.append(measured_sensordata)

  res_array = np.stack(all_residuals, axis=0)
  if initial_ndim == 1:
    res_array = res_array.ravel()
  else:
    res_array = res_array.reshape(res_array.shape[0], -1)

  return res_array.T, pred_sensordatas, measured_sensordatas


# Top level residual function
def build_residual_fn(**captured_kwargs):
  def built_residual_fn(x, params, **kwargs):
    return residual(
        x,
        params,
        **captured_kwargs,
        **kwargs,
    )
  return built_residual_fn

def residual(
    x: np.ndarray,
    params: parameter.ParameterDict,
    models_sequences: List[ModelSequences],
    build_model: Optional[BuildModelFn] = model_modifier.apply_param_modifiers,
    modify_residual: Optional[ModifyResidualFn] = None,
    custom_rollout: Optional[CustomRolloutFn] = None,
    n_threads: int = os.cpu_count(),
    return_pred_all: bool = False,
    resample_true: bool = True,
    sensor_weights: Optional[Mapping[str, float]] = None,
    enabled_observations: List[Tuple[str, timeseries.SignalType]] = []
):

  residuals = []
  preds = []
  records = []
  for model_sequences in models_sequences:
    for measured_rollout in model_sequences.measured_rollout:
      res = model_residual(
          x,
          params,
          lambda p: build_model(p, model_sequences.spec),
          measured_rollout,
          modify_residual,
          custom_rollout,
          n_threads,
          return_pred_all,
          resample_true,
          sensor_weights,
          enabled_observations
      )
      if isinstance(res, np.ndarray):
        residuals.append(res)
      else:
        residuals.append(res[0])
        preds.append(res[1])
        records.append(res[2])

  return residuals, preds, records


def _scipy_least_squares(
    x0: np.ndarray,
    residual_fn: Callable,
    bounds: np.ndarray,
    use_mujoco_jac: bool = False,
    **kwargs,
) -> scipy_optimize.OptimizeResult:
  max_nfev = kwargs.pop("max_iters", 200)
  if kwargs.pop("verbose", True):
    verbose = 2
  else:
    verbose = 0
  x_scale = kwargs.pop("x_scale", "jac")
  loss = kwargs.pop("loss", "linear")

  if use_mujoco_jac:
    # This is the default step sized for finite difference used in
    # scipy's least_sqaure and mujoco's minimize finite difference
    # https://github.com/scipy/scipy/blob/91e18f3bd355477b8b7747ec82d70ac98ffd2422/scipy/optimize/_numdiff.py#L404
    eps = np.finfo(np.float64).eps ** 0.5
    if "diff_step" in kwargs:
      eps = kwargs.pop("diff_step")

    def jac(x):
      return mujoco_minimize.jacobian_fd(
          residual=residual_fn,
          x=x.reshape((-1, 1)),
          r=residual_fn(x).reshape((-1, 1)),
          eps=eps,
          n_res=0,
          bounds=(bounds[0].reshape((-1, 1)), bounds[1].reshape((-1, 1))),
      )[0]

  else:
    jac = "2-point"

  return scipy_optimize.least_squares(
      residual_fn,
      x0,
      bounds=bounds,
      max_nfev=max_nfev,
      verbose=verbose,
      x_scale=x_scale,
      loss=loss,
      jac=jac,
      **kwargs,
  )


def _mujoco_least_squares(
    x0: np.ndarray,
    residual_fn: Callable,
    bounds: np.ndarray,
    **kwargs,
) -> scipy_optimize.OptimizeResult:
  if kwargs.pop("verbose", True):
    verbose = mujoco_minimize.Verbosity.FULLITER
  else:
    verbose = mujoco_minimize.Verbosity.SILENT
  max_iter = kwargs.pop("max_iters", 200)
  x, log = mujoco_minimize.least_squares(
      x0=x0,
      bounds=bounds,
      residual=residual_fn,
      verbose=verbose,
      max_iter=max_iter,
      **kwargs,
  )

  # If verbose, return the full optimization log.
  extras = {}
  if verbose == mujoco_minimize.Verbosity.FULLITER:
    extras["objective"] = [l.objective for l in log]
    extras["candidate"] = [l.candidate[:, 0] for l in log]

  return scipy_optimize.OptimizeResult(
      x=x,
      jac=log[-1].jacobian,
      grad=log[-1].grad,
      extras=extras,
  )


def optimize(
    initial_params: parameter.ParameterDict,
    residual_fn: Callable,
    optimizer: Literal[
        "scipy", "mujoco", "scipy_parallel_fd"
    ] = "mujoco",
    **optimizer_kwargs,
) -> scipy_optimize.OptimizeResult:

  x0 = initial_params.as_vector()
  bounds = initial_params.get_bounds()
  opt_params = initial_params.copy()

  # Check if there are any parameters to optimize.
  if len(opt_params) == 0 or opt_params.size == 0:
    print("The ParameterDict is empty or contains only frozen Parameters. Please declare all Parameters that need to be optimzied.")
    return opt_params, scipy_optimize.OptimizeResult(
        x=x0,
        jac=np.zeros((0, x0.shape[0])),
        grad=np.zeros_like(x0),
        extras={},
    )

  def optimized_residual_fn(x):
    residuals, _, _ = residual_fn(x, opt_params)
    return np.concatenate(residuals)

  if optimizer in ["scipy", "scipy_parallel_fd"]:
    opt_result =  _scipy_least_squares(
        x0,
        optimized_residual_fn,
        bounds,
        use_mujoco_jac=optimizer == "scipy_parallel_fd",
        **optimizer_kwargs,
    )
  elif optimizer == "mujoco":
    opt_result = _mujoco_least_squares(x0, optimized_residual_fn, bounds, **optimizer_kwargs)
  else:
    raise ValueError(f"Unsupported optimizer: '{optimizer}'. Expected one of: 'scipy', 'scipy_parallel_fd', or 'mujoco'.")

  # opt_params does not necessarily contain optimal parameter values
  # after the optimizer returns, so update it
  opt_params.update_from_vector(opt_result.x)

  return opt_params, opt_result



def calculate_intervals(
    residuals_star,
    J,
    alpha=0.05,
    lambda_zero_thresh=1e-15,
    v_zero_thresh=1e-8,
):
  if J.size == 0:
    return np.empty((0, 0)), np.empty((0,))

  # TODO(levi): account for per sensor variance
  # Estimate sensor variance by assuming a good model fit, so
  # remaining variance in the residual is due to sensor noise.
  # Dividing by n - p is an unbiased estimate of the noise.
  final_r = np.concatenate(residuals_star)
  s2 = np.dot(final_r, final_r) / (final_r.size - J.shape[1])
  H = J.T @ J

  # Calculate the diagonals of the inverse of H
  # using the observation that division by zero
  # of eig(H) close to zero is canceled by numerically
  # zero elements of the eigenvectors
  # That is numerically zero eigenvalues only
  # cause a confidence bound to be infinite if that eigenvalue
  # has a numerically non-zero effect on the considered parameter
  lamb, V = np.linalg.eigh(H)
  lamb_max = np.max(lamb)
  diag_inv_H = []
  for j in range(H.shape[0]):
    inv_H_jj = 0.0
    v_j_max = np.max(np.abs(V[:, j]))
    for i in range(H.shape[0]):
      lambda_i = lamb[i]
      if lambda_i / lamb_max < lambda_zero_thresh:
        lambda_i = 0.0

      v_j_i = V[j, i]
      if np.abs(v_j_i / v_j_max) < v_zero_thresh:
        v_j_i = 0.0

      if lambda_i == 0.0 and v_j_i != 0.0:
        inv_H_jj += np.inf
      elif lambda_i == 0.0 and v_j_i == 0.0:
        pass
      else:
        inv_H_jj += v_j_i**2 / lambda_i
    diag_inv_H.append(inv_H_jj)
  diag_inv_H = np.array(diag_inv_H)

  # In general eigenvalue decomposition should be more accurate
  # than calculating the inverse of H using a general method
  # TODO(levi): expand the eigenvalue/eigenvector element cancelation above to the full inverse matrix
  # inv_H = V @ np.diag(np.divide(1, lamb, out=np.inf*np.zeros_like(lamb), where=lamb != 0.0)) @ V.T
  lamb[lamb == 0] = lambda_zero_thresh
  inv_H = V @ np.diag(1 / lamb) @ V.T
  # print('inv test')
  # print(np.diag(inv_H @ H))
  # print(np.diag(np.linalg.inv(H) @ H)))
  Sigma_X = s2 * inv_H
  intervals = np.sqrt(diag_inv_H * s2) * stdtrit(
      final_r.size - J.shape[1], 1 - alpha / 2
  )
  return Sigma_X, intervals


def save_results(
    experiment_results_folder: os.PathLike,
    models_sequences: Sequence[ModelSequences],
    initial_params: parameter.ParameterDict,
    opt_params: parameter.ParameterDict,
    opt_result: scipy_optimize.OptimizeResult,
    residual_fn,
):
  experiment_results_folder = pathlib.Path(experiment_results_folder)
  if not experiment_results_folder.exists():
    experiment_results_folder.mkdir(parents=True, exist_ok=True)
  logging.info(
      "Experiment results will be saved to %s", experiment_results_folder
  )

  ####################################
  # Save optimization results to disk
  ####################################

  initial_params.save_to_disk(experiment_results_folder / "params_x_0.yaml")
  opt_params.save_to_disk(experiment_results_folder / "params_x_hat.yaml")

  with open(
      os.path.join(experiment_results_folder, "results.pkl"), "wb"
  ) as handle:
    pickle.dump(opt_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # TODO these intervals should be part of the params object
  residuals_star, _, _ = residual_fn(opt_result.x, opt_params, return_pred_all=True)
  covariance, intervals = calculate_intervals(residuals_star, opt_result.jac)
  with open(
      os.path.join(experiment_results_folder, "confidence.pkl"), "wb"
  ) as handle:
    pickle.dump(
        {"cov": covariance, "intervals": intervals},
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


  # Save the ID'd models out
  for model_sequences in models_sequences:
    model_sequences.spec.to_file(
        (experiment_results_folder / f"{model_sequences.name}.xml").as_posix()
    )

  # Print out nominal compared to initial
  print("Initial Parameters")
  x0 = initial_params.as_vector()
  x_nominal = initial_params.as_nominal_vector()
  print(initial_params.compare_parameters(x0, opt_result.x, measured_params=x_nominal))


def default_report(
    models_sequences: Sequence[ModelSequences],
    initial_params: parameter.ParameterDict,
    opt_params: parameter.ParameterDict,
    residual_fn,
    opt_result: scipy_optimize.OptimizeResult,
    title="SysID",
    save_path=None,
    build_model: Optional[BuildModelFn] = model_modifier.apply_param_modifiers,
    generate_videos=True,

) -> ReportBuilder:

  """Returns a ReportBuilder containing experiment results.

  Users needing a custom report can copy and modify this code.
  """
  from mujoco.sysid.report.sections.video import VideoPlayer, generate_video_from_trajectory
  from mujoco.sysid.report.sections.group import GroupSection
  from mujoco.sysid.report.sections.row import RowSection
  from mujoco.sysid.report.sections.parameter_distribution import ParameterDistribution
  from mujoco.sysid.report.sections.insights import AutomatedInsights

  x0 = initial_params.as_vector()
  x_nominal = initial_params.as_nominal_vector()
  x_hat = opt_params.as_vector()

  ####################################
  # Build report
  # Sections:
  # Fit
  # Parameter tables
  # Confidence intervals
  # Extras: Optimization trace
  ####################################
  rb = ReportBuilder(title)

  if generate_videos:
    # 1. Video Player
    model_sequences_to_render = models_sequences[0]
    traj_to_render = model_sequences_to_render.measured_rollout[0]
    model_spec_to_render = model_sequences_to_render.spec

    video_dir = pathlib.Path(save_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Video 1: All
    video_all_path = video_dir / "video_all.mp4"
    generate_video_from_trajectory(
        initial_params=initial_params,
        opt_params=opt_params,
        build_model=build_model,
        traj_measured=traj_to_render,
        model_spec=model_spec_to_render,
        output_filepath=video_all_path,
      )

    # Video 2: Initial + Nominal
    video_init_path = video_dir / "video_init.mp4"
    generate_video_from_trajectory(
        initial_params=initial_params,
        opt_params=opt_params,
        build_model=build_model,
        traj_measured=traj_to_render,
        model_spec=model_spec_to_render,
        output_filepath=video_init_path,
        render_opt=False,
      )

    # Video 3: Optimized + Nominal
    video_opt_path = video_dir / "video_opt.mp4"
    generate_video_from_trajectory(
        initial_params=initial_params,
        opt_params=opt_params,
        build_model=build_model,
        traj_measured=traj_to_render,
        model_spec=model_spec_to_render,
        output_filepath=video_opt_path,
        render_initial=False,
      )

    video_all_section = VideoPlayer(
        title="All Models",
        video_filepath=video_all_path,
        anchor="visual_run_all",
        autoplay=True,
        muted=True,
        width="100%",
        height=None,
        caption="<span style='color:red'>Red=Initial</span>, <span style='color:green'>Green=Nominal</span>, <span style='color:blue'>Blue=Optimized</span>"
    )
    
    video_init_section = VideoPlayer(
        title="Initial vs Nominal",
        video_filepath=video_init_path,
        anchor="visual_run_init",
        autoplay=True,
        muted=True,
        width="100%",
        height=None,
        caption="<span style='color:red'>Red=Initial</span>, <span style='color:green'>Green=Nominal</span>"
    )

    video_opt_section = VideoPlayer(
        title="Optimized vs Nominal",
        video_filepath=video_opt_path,
        anchor="visual_run_opt",
        autoplay=True,
        muted=True,
        width="100%",
        height=None,
        caption="<span style='color:green'>Green=Nominal</span>, <span style='color:blue'>Blue=Optimized</span>"
    )

    rb.add_section(RowSection(
        title="Visual Comparison",
        sections=[video_all_section, video_init_section, video_opt_section],
        anchor="visual_comparison",
        description="Visual comparison of the system identification results. The nominal model is shown in green, the initial model in red, and the optimized model in blue."
    ))

  # 2. Automated Insights (Logs)
  rb.add_section(AutomatedInsights("Automated Insights", opt_params))

  # 3. Parameters Table (Unified)
  rb.add_section(ParametersTable("Parameters", opt_params, initial_params, anchor="Parameters"))

  # 4. Control Signals
  # Get predictions for initial solution.
  names = [
      f"{model_sequences.name}\n{sequence}"
      for model_sequences in models_sequences
      for sequence in model_sequences.sequence_name
  ]
  _, pred0s, _ = residual_fn(initial_params.as_vector(), initial_params, return_pred_all=True)

  residuals_star, preds_star, records_star = residual_fn(
      opt_params.as_vector(), opt_params, return_pred_all=True
  )

  model_hat = build_model(initial_params, models_sequences[0].spec)
  ctrl_ts = models_sequences[0].control
  rb.add_section(SignalReport("Control Signals", model_hat, title_prefix=f"", ts_dict={"signal":ctrl_ts[0]}, anchor="control_signals"))

  # 5. Observation Signals
  observation_reports = []
  for name, pred, record, pred0 in zip(names, preds_star, records_star, pred0s):
    obs_dict = {
      "initial": pred0[0],
      "nominal": record[0],
      "fitted": pred[0]
    }
    observation_reports.append(SignalReport(f"Sequence: {name}", model_hat, title_prefix=f"",ts_dict=obs_dict, collapsible=True))
  
  rb.add_section(GroupSection("Observation Signals", observation_reports, anchor="observations"))

  covariance, intervals = calculate_intervals(
      residuals_star, opt_result.jac
  )

  # 6. Parameter Distribution
  rb.add_section(ParameterDistribution(
      title="Parameter Distribution",
      opt_params=opt_params,
      initial_params=initial_params,
      confidence_intervals=intervals,
      anchor="param_dist"
  ))

  rb.add_section(Covariance(
          title="Covariance and Correlation",
          anchor="cov",
          covariance=covariance,
          parameter_dict=opt_params,
  ))

  # Add diagnostic optimization trace plots.
  if "extras" in opt_result:
    # Add to the report.
    rb.add_section(
      OptimizationTrace(
                  title="Optimization Trace",
                  anchor="opt",
                  objective=opt_result.extras.get("objective"),
                  candidate=opt_result.extras.get("candidate"),
                  bounds=opt_params.get_bounds(),
                  param_names=opt_params.get_non_frozen_parameter_names(),
              )
        )

  rb.build()
  if save_path:
    rb.save(save_path / "report.html")
  return rb


# TODO(nimrod): Consider deleting this function, given we can export plots from
#  plotly either on the web or with fig.write_image.
def default_report_matplotlib(
    experiment_results_folder: os.PathLike,
    models_sequences: Sequence[ModelSequences],
    params: parameter.ParameterDict,
    sysid_residual,
    x0: np.ndarray,
    opt_result: scipy_optimize.OptimizeResult,
    build_model: Optional[BuildModelFn] = model_modifier.apply_param_modifiers,
):
  """Outputs PNG plots to the experiment results folder."""
  experiment_results_folder = pathlib.Path(experiment_results_folder)
  if not experiment_results_folder.exists():
    experiment_results_folder.mkdir(parents=True, exist_ok=True)

  x_hat = opt_result.x
  params.update_from_vector(x_hat)

  # Save the ID'd models out
  for model_sequences in models_sequences:
    model_hat = build_model(params, model_sequences.spec)

  # Get predictions for initial solution.
  params.update_from_vector(x0)
  names = [
      f"{model_sequences.name}\n{sequence}"
      for model_sequences in models_sequences
      for sequence in model_sequences.sequence_name
  ]
  _, pred0s, record0s = sysid_residual(x0, return_pred_all=True)

  for name, pred0, record0 in zip(names, pred0s, record0s):
    plotting.plot_sensor_comparison(
        model_hat,
        predicted_times=pred0[0].times,
        predicted_data=pred0[0].data,
        real_times=record0[0].times,
        real_data=record0[0].data,
        title_prefix=f"x0 {name}",
        size_factor=0.5,
    )
    name_fig = name.replace("/", " ")
    name_fig = name_fig.replace("\n", " ")
    plt.savefig(os.path.join(experiment_results_folder, f"x0 {name_fig}.png"))

  residuals_star, preds_star, records_star = sysid_residual(
      x_hat, return_pred_all=True
  )
  for name, pred, record, pred0 in zip(names, preds_star, records_star, pred0s):
    plotting.plot_sensor_comparison(
        model_hat,
        predicted_times=pred[0].times,
        predicted_data=pred[0].data,
        real_times=record[0].times,
        real_data=record[0].data,
        title_prefix=f"x* {name}",
        size_factor=0.5,
    )
    name_fig = name.replace("/", " ")
    name_fig = name_fig.replace("\n", " ")
    plt.savefig(experiment_results_folder / f"xstar {name_fig}.png")

  # Add diagnostic optimization trace plots.
  if "extras" in opt_result:
    # Objective value over iterations.
    objective = opt_result.extras["objective"]
    plotting.plot_objective(objective)
    plt.savefig(experiment_results_folder / "loss.png", dpi=300)

    # Candidate parameter values over iterations.
    candidate = opt_result.extras["candidate"]

    # Candidate parameter values over iterations.
    # Candidate heatmap over iterations.
    plotting.plot_candidate_heatmap(
        candidate,
        param_names=params.get_non_frozen_parameter_names(),
        bounds=params.get_bounds(),
    )
    plt.savefig(experiment_results_folder / "candidate_heatmap.png", dpi=300)

    plotting.plot_candidate(
        candidate,
        bounds=params.get_bounds(),
        param_names=params.get_non_frozen_parameter_names(),
    )
    plt.savefig(experiment_results_folder / "candidate.png", dpi=300)

  _, intervals = calculate_intervals(residuals_star, opt_result.jac)
  plotting.parameter_confidence(
      all_exp_names=["trial"], all_params=[params], all_intervals=[intervals]
  )
  #   plotting.parameter_confidence(["trial"], [params], [x_hat], [intervals])
  plt.savefig(experiment_results_folder / "params.png")
