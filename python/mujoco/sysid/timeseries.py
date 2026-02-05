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
"""Time series utilities."""

from dataclasses import dataclass
import pathlib
from typing import Dict, Literal, Optional, Tuple, Union, List, TypeAlias

import numpy as np
import scipy.interpolate
import mujoco
from enum import Enum


class SignalType(Enum):
  MjSensor = 0
  CustomObs = 1
  MjStateQPos = 2
  MjStateQVel = 3
  MjStateAct = 4
  MjCtrl = 5

SignalMappingType: TypeAlias = dict[str, Tuple[SignalType, np.ndarray]]

InterpolationMethod = Literal[
    "linear", "cubic", "quadratic", "quintic", "zero_order_hold", "zoh"
]


def get_ctrl_map(model: mujoco.MjModel) -> SignalMappingType:
    """
    Created a dictionary mapping from a MuJoCo models ctrl names to indices

    Args:
        model: The loaded mujoco.MjModel object.

    Returns:
        A dictionary where keys are ctrl names and values are
        lists of ctrl indices
    """
    ctrl_map: SignalMappingType = {}

    for act_id in range(model.nu):
        act_name = model.actuator(act_id).name
        ctrl_indices = np.arange(act_id, act_id + 1)
        ctrl_map[f"{act_name}_ctrl"] = (SignalType.MjCtrl, ctrl_indices)

    return ctrl_map


def get_state_maps(model: mujoco.MjModel) -> Tuple[SignalMappingType, SignalMappingType, SignalMappingType, SignalMappingType]:
    """
    Creates  dictionary mappings for qpos, qvel, ctrl, and act
    """
    qpos_map: SignalMappingType = {}
    qvel_map: SignalMappingType = {}
    act_map: SignalMappingType = {}
    ctrl_map: SignalMappingType = {}

    nq = model.nq
    nv = model.nv
    for body_id in range(model.nbody):
        b = model.body(body_id)
        body_name = b.name
        start_index = model.body_dofadr[body_id]

        if start_index >= 0 and b.dofnum[0] == 6:
            qpos_indices = np.arange(start_index, start_index + 7)
            qpos_map[f"{body_name}_qpos"] = (SignalType.MjStateQPos, qpos_indices)
            qvel_indices = np.arange(start_index+nq, start_index + nq + 6)
            qvel_map[f"{body_name}_qvel"] = (SignalType.MjStateQVel, qvel_indices)

    for jnt_id in range(model.njnt):
        jnt_name = model.joint(jnt_id).name
        start_index = model.jnt_qposadr[jnt_id]
        jnt_type = model.jnt_type[jnt_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            qpos_width= 1
            qvel_width= 1
        elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
            qpos_width= 1
            qvel_width= 1
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_width= 4
            qvel_width= 3
        elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        else:
            raise ValueError(f"Warning: Unknown joint type {jnt_type} for joint {jnt_name}")

        qpos_indices = np.arange(start_index, start_index + qpos_width)
        qpos_map[f"{jnt_name}_qpos"] = (SignalType.MjStateQPos, qpos_indices)
        qvel_indices = np.arange(start_index + nq, start_index + nq + qvel_width)
        qvel_map[f"{jnt_name}_qvel"] = (SignalType.MjStateQVel, qvel_indices)


    for act_id in range(model.nu):
        act_name = model.actuator(act_id).name
        start_index = model.actuator_actadr[act_id]
        num_vals = model.actuator_actnum[act_id]
        indices = np.arange(start_index + nq + nv, start_index + nq + nv + num_vals)

        # if index is -1, the actuator is stateless. Ignore.
        if start_index != -1:
            act_map[f"{act_name}_act"] = (SignalType.MjStateAct, indices)

        ctrl_indices = np.arange(act_id, act_id + 1)
        ctrl_map[f"{act_name}_ctrl"] = (SignalType.MjCtrl, ctrl_indices)

    total_mapped_indices_qpos = 0
    for _,indices in qpos_map.values():
        total_mapped_indices_qpos += len(indices)

    total_mapped_indices_qvel = 0
    for _,indices in qvel_map.values():
        total_mapped_indices_qvel += len(indices)

    if total_mapped_indices_qpos != model.nq or total_mapped_indices_qvel != model.nv:
        raise ValueError("The number of mapped indcies does not match the number of states.")

    return qpos_map, qvel_map, act_map, ctrl_map


def mj_sensor_to_map(model: mujoco.MjModel) -> SignalMappingType:
  """Constructs an signal_mapping dict from MuJoCo sensors.

  Args:
    model: MuJoCo model

  Returns:
    signal_mapping: Dict of observation(sensor) name to its indices in .data
  """
  signal_mapping = {}
  for sensor_id in range(model.nsensor):
    addr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
    indices = np.arange(addr, addr + dim)
    signal_mapping[name] = (SignalType.MjSensor, indices)

  return signal_mapping


@dataclass(frozen=True)
class TimeSeries:
  """A utility class for working with time-series data.

  Attributes:
    times: 1D array of timestamps.
    data: Array of signal data. The first axis corresponds to time.
    signal_mapping: Dict of tuples that maps the signal type and its
    signal fields in data
  """

  times: np.ndarray
  data: np.ndarray
  signal_mapping: Optional[Dict[str, Tuple[SignalType, np.ndarray]]] = None

  def get_indices(self, obs_name):
    if obs_name not in self.signal_mapping:
      raise ValueError(f"{obs_name} observation is not in the observation name map.")
    return self.signal_mapping[obs_name]

  @classmethod
  def create(
    cls,
    times: np.ndarray,
    data: np.ndarray,
    signal_mapping: Dict[str, Tuple[SignalType, Union[List, int]]]
   ) -> "TimeSeries":
      for key in signal_mapping:
        signal_type, indices = signal_mapping[key]
        indices = np.atleast_1d(indices)
        signal_mapping[key] = (signal_type, indices)

      return cls(times, data, signal_mapping)


  @classmethod
  def slice_by_name(
      cls,
      ts: 'TimeSeries',
      enabled_sensors: List[str]
  ) -> 'TimeSeries':

      if not ts.signal_mapping:
        return ts

      original_indices_to_keep = []
      original_to_new_index_map = {}
      all_original_indices = []
      for name in ts.signal_mapping:
          all_original_indices.extend(ts.signal_mapping[name][1])

      # Build a set of indices to keep for quick lookups
      kept_indices_set = set()
      for name in enabled_sensors:
          if name not in ts.signal_mapping:
              raise ValueError(f"Attemping to slice TimeSeries failed. {name} is not in {ts.signal_mapping}.")
          kept_indices_set.update(ts.signal_mapping[name][1])
      new_index_counter = 0
      for original_index in all_original_indices:
          if original_index in kept_indices_set:
              original_to_new_index_map[original_index] = new_index_counter
              new_index_counter += 1
              original_indices_to_keep.append(original_index)


      data = ts.data[..., original_indices_to_keep]

      trimmed_signal_mapping = {}
      for name in enabled_sensors:
          metadata, original_indices = ts.signal_mapping[name]

          new_indices = []
          for original_index in original_indices:
              new_indices.append(original_to_new_index_map[original_index])

          trimmed_signal_mapping[name] = (metadata, np.asarray(new_indices))

      return cls(ts.times, data, trimmed_signal_mapping)


  @classmethod
  def from_arrays(
    cls,
    times: np.ndarray,
    data: np.ndarray,
    names: List[str]
    ) -> 'TimeSeries':
    """
    Constructs a TimeSeries object from a single stacked data array, len(names) is assumed to equal the number of columns in data.
    Assumes 'data' is a 2D array of shape (N, D), where N is time steps
    and D is the total number of observations.
    """
    if data.ndim != 2:
        raise ValueError("The 'data' array must be 2-dimensional (Time x Features).")

    num_total_dims = data.shape[1]
    if num_total_dims != len(names):
        raise ValueError(
            f"The number of column names ({len(names)}) must match the "
            f"number of columns in the data array ({num_total_dims})."
        )

    signal_mapping_dict: Dict[str, np.ndarray] = {
        name: np.array([i])
        for i, name in enumerate(names)
    }

    return cls(
        times=times,
        data=data,
        signal_mapping=signal_mapping_dict
    )


  @classmethod
  def from_list(
      cls,
      times: np.ndarray,
      data: List[np.ndarray],
      names: List[str]
  ) -> 'TimeSeries':
      """
      Constructs a TimeSeries object from pre-separated sensor data arrays.

      'data' is a list where each item is a NumPy array corresponding to a single
      sensor's measurements over time.
      """
      if len(data) != len(names):
          raise ValueError(
              f"The number of sensor data arrays ({len(data)}) must match the "
              f"number of sensor names ({len(names)})."
          )

      if not all(d.ndim == 1 or d.ndim == 2 for d in data):
          raise ValueError("All arrays in 'data' must be 1D (for a single measurement) or 2D.")

      n_samples = times.shape[0]
      if not all(d.shape[0] == n_samples for d in data):
           raise ValueError("All sensor data arrays must have the same number of samples as 'times'.")

      signal_mapping_dict: Dict[str, np.ndarray] = {}
      current_col_index = 0
      stacked_data_list = []

      for sensor_data, name in zip(data, names):
          if sensor_data.ndim == 1:
              sensor_data = sensor_data[:, np.newaxis]
          n_cols = sensor_data.shape[1]
          indices = np.arange(current_col_index, current_col_index + n_cols)
          signal_mapping_dict[name] = indices
          current_col_index += n_cols
          stacked_data_list.append(sensor_data)
      data_stacked = np.hstack(stacked_data_list)

      return cls(
          times=times,
          data=data_stacked,
          signal_mapping=signal_mapping_dict
      )

  @classmethod
  def from_mj_sensordata(
    cls,
    times: np.ndarray,
    data: np.ndarray,
    names: List[str],
    model: mujoco.MjModel
    ) -> 'TimeSeries':
    """
    Constructs a TimeSeries object from MuJoCo sensors.
    """

    if data.shape[1] != len(names):
      raise ValueError(f"The number of observation names ({len(names)}) does not match the number of columns of data ({data.shape[1]}).")

    signal_mapping = mj_sensor_to_map(model)

    return cls(
      times=times,
      data=data,
      signal_mapping=signal_mapping
    )

  @classmethod
  def from_mj_state(
    cls,
    times: np.ndarray,
    data: list[np.ndarray],
    names: List[Tuple[str, SignalType]],
    model: mujoco.MjModel
    ) -> 'TimeSeries':
    """
    Constructs a TimeSeries object from MuJoCo data.
    This can either be MuJoCo state observations or MuJoCo sensor data.
    Attributes:
        times (np.ndarray): A 1D array of timestamps for each observation.
        data (list[np.ndarray]): A list of NumPy arrays, each containing the
            time-series data for a single observation or sensor.
        names (List[Tuple[str, SignalType]]): A list of (name, type) tuples
            describing the content of the `data` arrays.
        model (mujoco.MjModel): The MuJoCo model used to generate the data.

    Returns:
        TimeSeries: A new TimeSeries object.
    """
    if len(data) != len(names):
      raise ValueError(f"The number of observation names ({len(names)}) does not match the number of columns of data ({len(data)}).")

    new_signal_mapping = {}
    stacked_data_list = []
    current_col_index = 0

    index = 0

    for sensor_data, obs_info in zip(data, names):
        name, obs_type = obs_info
        if sensor_data.ndim == 1:
            sensor_data = sensor_data[:, np.newaxis]
        n_cols = sensor_data.shape[1]
        indices = np.arange(current_col_index, current_col_index + n_cols)
        current_col_index += n_cols
        stacked_data_list.append(sensor_data)

        indices = np.arange(index, index + sensor_data.shape[-1])
        index += sensor_data.shape[-1]
        if obs_type == SignalType.MjStateQPos:
            name = f"{name}_qpos"
        elif obs_type == SignalType.MjStateQVel:
            name = f"{name}_qvel"
        elif obs_type == SignalType.MjStateAct:
            name = f"{name}_act"

        new_signal_mapping[name] = (obs_type, np.asarray(indices))
    data_stacked = np.hstack(stacked_data_list)

    return cls(
      times=times,
      data=data_stacked,
      signal_mapping=new_signal_mapping
    )

  def __post_init__(self):
    """Validate the time series data after initialization.

    Raises:
      ValueError: If times is not 1D, if lengths don't match, if times
        is not strictly increasing, or if arrays are empty.
    """
    if self.times.size == 0 or self.data.size == 0:
      raise ValueError("Empty arrays are not allowed in TimeSeries")
    if self.times.ndim != 1:
      raise ValueError(
          f"times must be a 1D array, got {self.times.ndim}D array"
      )
    if len(self.times) != len(self.data):
      raise ValueError(
          f"Length of times ({len(self.times)}) and data ({len(self.data)})"
          " must match"
      )
    if not np.all(np.diff(self.times) > 0):
      raise ValueError("times must be strictly increasing")

  def __len__(self) -> int:
    return len(self.data)

  def save_to_disk(self, path: pathlib.Path) -> None:
    """Save the time series data to disk.

    Args:
      path: Path where the data will be saved.
    """
    np.savez(path, times=self.times, data=self.data, signal_mapping=np.array(self.signal_mapping, dtype=object))

  def save_to_csv(self, path: pathlib.Path) -> None:
    """Save the time series data to a CSV file."""
    np.savetxt(
        path,
        np.concatenate([self.times[:, None], self.data], axis=1),
        delimiter=",",
    )

  @classmethod
  def load_from_disk(cls, path: pathlib.Path) -> "TimeSeries":
    """Load time series data from disk.

    Args:
      path: Path to the saved data.

    Returns:
      A new TimeSeries object.
    """
    with np.load(path, allow_pickle=True) as npz:
      times = npz["times"]
      data = npz["data"]
      if "signal_mapping" in npz:
        signal_mapping = npz["signal_mapping"].item()
      else:
        signal_mapping = None

    return cls(times=times, data=data, signal_mapping=signal_mapping)

  def interpolate(
      self, t: Union[float, np.ndarray], method: InterpolationMethod = "linear"
  ) -> np.ndarray:
    """Interpolate data at specified time(s).

    This is the core interpolation function used by both get() and resample().

    Args:
      t: Time point(s) at which to interpolate data.
      method: Interpolation method to use.

    Returns:
      Interpolated data values.
    """
    t = np.atleast_1d(np.asarray(t))

    if method in ("zero_order_hold", "zoh"):
      indices = np.searchsorted(self.times, t, side="right") - 1
      indices = np.clip(indices, 0, len(self.times) - 1)
      return self.data[indices]

    return scipy.interpolate.interp1d(
        self.times,
        self.data,
        kind=method,
        axis=0,
        bounds_error=False,
        fill_value=(self.data[0], self.data[-1]),
        assume_sorted=True,
    )(t)

  def get(
      self, t: Union[float, np.ndarray], method: InterpolationMethod = "linear"
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Get interpolated data at specified time(s).

    This method is useful for querying data at specific timestamps without
    creating a new TimeSeries object.

    Args:
      t: Time point(s) at which to get data.
      method: Interpolation method to use.

    Returns:
      Tuple of (times, interpolated_data).
    """
    t_orig = np.asarray(t)
    t_shape = t_orig.shape
    result = self.interpolate(t_orig, method=method)
    if t_shape == ():
      result = result.squeeze(axis=0)
    return t_orig, result

  def resample(
      self,
      new_times: Optional[np.ndarray] = None,
      target_dt: Optional[float] = None,
      method: InterpolationMethod = "linear",
  ) -> "TimeSeries":
    """Resample the time series to new timestamps or a specific time interval.

    This method creates a new TimeSeries object with data interpolated at the
    specified timestamps.

    Args:
      new_times: Optional array of new timestamps. If provided, target_dt is
        ignored.
      target_dt: Optional time interval for regular resampling. Only used if
        new_times is None.
      method: Interpolation method to use.

    Returns:
      A new TimeSeries object with resampled data.

    Raises:
      ValueError: If neither new_times nor target_dt is provided, or if
        new_times is not strictly increasing.
    """
    # Generate new times if target_dt is provided.
    if new_times is None:
      if target_dt is None:
        raise ValueError("Either new_times or target_dt must be provided")
      if target_dt <= 0:
        raise ValueError("target_dt must be a positive float")

      # Create evenly spaced timestamps.
      new_nsteps = (
          int(np.ceil((self.times[-1] - self.times[0]) / target_dt)) + 1
      )
      new_times = np.linspace(
          self.times[0], self.times[-1], new_nsteps, endpoint=True
      )
    else:
      # Make sure new_times is valid.
      if new_times.ndim != 1:
        raise ValueError("new_times must be a 1D array")
      if not np.all(np.diff(new_times) > 0):
        raise ValueError("new_times must be strictly increasing")

    new_data = self.interpolate(new_times, method=method)
    return TimeSeries(times=new_times, data=new_data, signal_mapping=self.signal_mapping)

  def remove_from_beginning(self, time_to_remove_s: float) -> "TimeSeries":
    """Remove time from the beginning of the time series.

    Args:
      time_to_remove_s: Time to remove from the beginning of the time series.

    Returns:
      A new TimeSeries object with the specified time removed.
    """
    if time_to_remove_s < 0:
      raise ValueError("time_to_remove_s must be non-negative")
    if time_to_remove_s > self.times[-1]:
      raise ValueError(
          "time_to_remove_s is greater than the duration of the time series"
      )
    idx = np.searchsorted(self.times, time_to_remove_s)
    times_shifted = self.times[idx:] - self.times[idx]
    return TimeSeries(times=times_shifted, data=self.data[idx:], signal_mapping=self.signal_mapping)

  def dt_statistics(self) -> Dict[str, float]:
    """Calculate statistics about the time intervals.

    Returns:
      Dictionary with mean, median, std, min, and max of time intervals.

    Raises:
      ValueError: If there are fewer than two timestamps.
    """
    if self.times.size < 2:
      raise ValueError(
          "Must have at least two timestamps to compute dt statistics."
      )
    dt_values = np.diff(self.times)
    stats = {}
    for fn in ["mean", "median", "std", "min", "max"]:
      stats[fn] = float(getattr(np, fn)(dt_values))
    return stats

  def __repr__(self) -> str:
    """Return a string representation of the TimeSeries object."""
    t_start, t_end = self.times[0], self.times[-1]
    duration = t_end - t_start

    data_shape = self.data.shape
    n_samples = len(self)

    dt_stats = self.dt_statistics()
    mean_dt = dt_stats["mean"]
    min_dt = dt_stats["min"]
    max_dt = dt_stats["max"]

    is_uniform = dt_stats["std"] / mean_dt < 0.01  # Less than 1% variation.

    # Calculate data range (min/max values).
    data_min = np.min(self.data)
    data_max = np.max(self.data)
    data_range = f"[{data_min:.3g}, {data_max:.3g}]"

    parts = [
        f"TimeSeries(",
        f"  samples={n_samples}",
        f"  shape={data_shape}",
        f"  time_range=[{t_start:.3g}, {t_end:.3g}] (duration={duration:.3g})",
        f"  dt={mean_dt:.3g}"
        + (
            " (uniform)"
            if is_uniform
            else f" (min={min_dt:.3g}, max={max_dt:.3g})"
        ),
        f"  data_range={data_range}",
        f"  signal_mapping={self.signal_mapping}",
        f")",
    ]

    return "\n".join(parts)
