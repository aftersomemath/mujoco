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
"""Common signal modifiers."""

from typing import Dict, List, Mapping, Optional, Union

import mujoco
import numpy as np

from mujoco_sysid import parameter
from mujoco_sysid import timeseries


def _get_sensor_indices(model: mujoco.MjModel, sensor_name: str) -> List[int]:
  sensor_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_SENSOR.value, sensor_name
  )
  if sensor_id == -1:
    raise ValueError(f"Sensor not found in model: {sensor_name}")

  addr = model.sensor_adr[sensor_id]
  dim = model.sensor_dim[sensor_id]

  return list(range(addr, addr + dim))


def get_sensor_indices(
    model: mujoco.MjModel,
    sensor_name: Union[str, List[str]],
    sort: bool = False,
) -> List[int]:
  """Get sensor indices from a sensor configuration dictionary.

  Args:
      model: MuJoCo model containing the sensors.
      sensor_name: sensor name or list of names to return the indices for
  """
  if isinstance(sensor_name, str):
    return _get_sensor_indices(model, sensor_name)
  all_indices = []
  for name in sensor_name:
    all_indices.extend(_get_sensor_indices(model, name))
  if sort:
    return sorted(all_indices)
  return all_indices


def apply_bias(
    ts: timeseries.TimeSeries,
    model: mujoco.MjModel,
    sensor_name: str,
    bias: parameter.Parameter,
) -> timeseries.TimeSeries:
  indices = ts.get_indices(sensor_name)[1]
  data_out = ts.data.copy()
  data_out[..., indices] += bias.value
  return timeseries.TimeSeries(ts.times, data_out, ts.signal_mapping)


def apply_gain(
    ts: timeseries.TimeSeries,
    model: mujoco.MjModel,
    sensor_name: str,
    gain: parameter.Parameter,
) -> timeseries.TimeSeries:
  indices = ts.get_indices(sensor_name)[1]
  data_out = ts.data.copy()
  data_out[..., indices] *= gain.value
  return timeseries.TimeSeries(ts.times, data_out, ts.signal_mapping)


def apply_delay(
    ts: timeseries.TimeSeries,
    model: mujoco.MjModel,
    sensor_name: str,
    delay: parameter.Parameter,
) -> timeseries.TimeSeries:
  indices = ts.get_indices(sensor_name)[1]

  ts_sensor = timeseries.TimeSeries(ts.times, ts.data[:, indices],ts.signal_mapping)
  ts_sensor_delayed = ts_sensor.resample(ts.times - delay.value)

  ts_delayed = timeseries.TimeSeries(ts.times, ts.data, ts.signal_mapping)
  ts_delayed.data[:, indices] = ts_sensor_delayed.data

  return ts_delayed


def apply_time_window(
    ts: timeseries.TimeSeries,
    min_t: float,
    max_t: float,
) -> timeseries.TimeSeries:
  """Select a subset of a timeseries whose timestamps plus the max delay can be
  sampled from ts_sample."""
  min_i = np.searchsorted(ts.times, min_t, side="left")
  max_i = np.searchsorted(ts.times, max_t, side="right")
  return timeseries.TimeSeries(
      ts.times[min_i:max_i], ts.data[min_i:max_i], ts.signal_mapping
  )


def apply_delayed_ts_window(
    ts: timeseries.TimeSeries,
    ts_delayed: timeseries.TimeSeries,
    min_delay: float,
    max_delay: float,
) -> timeseries.TimeSeries:
  """Window a timeseries so that the included timestamps lay within the bounds of a
  timeseries that may be delayed between min_delay and max_delay.

  Args:
    ts: The timeseries to window.
    ts_delayed: The timeseries to use as the bounds.
    min_delay: The minimum delay. May be negative.
    max_delay: The maximum delay.

  Returns:
    A new timeseries with the timestamps windowed.
  """
  if min_delay > max_delay:
    raise ValueError(
        "min_delay must be less than or equal to max_delay, "
        f"received {min_delay} and {max_delay}"
    )
  return apply_time_window(
      ts, ts_delayed.times[0] - min_delay, ts_delayed.times[-1] - max_delay  )


def apply_resample_and_delay(
    ts: timeseries.TimeSeries,
    times: np.ndarray,
    default_delay: float,
    model: mujoco.MjModel,
    sensor_delays: Optional[Dict[str, float]] = None,
    predicted_data: bool = True,
) -> timeseries.TimeSeries:
  delays = [default_delay]*ts.data.shape[1]
  for name, delay in sensor_delays:
    sensor_indices = ts.get_indices(name)[1]
    for i in sensor_indices:
      delays[i] = delay

  # TODO does resampling each signal individually, in order to support
  # non-contiguous indices, have a significant performance hit?
  resampled_ts = []
  for i, d in enumerate(delays):
    if predicted_data: # TODO why is this needed?
      d = -d
    ts_sliced = timeseries.TimeSeries(ts.times, ts.data[:, i:i+1], ts.signal_mapping)
    ts_sliced_resampled = ts_sliced.resample(times + d)
    resampled_ts.append(ts_sliced_resampled)

  data_stacked = np.concatenate([ts.data for ts in resampled_ts], axis=1)
  return timeseries.TimeSeries(times, data_stacked, ts.signal_mapping)


def prepare_sensor_weights(
    sensor_weights: Mapping[str, float] | np.ndarray,
    n_sensors: int,
    model: mujoco.MjModel,
) -> np.ndarray:
  if isinstance(sensor_weights, np.ndarray):
    if sensor_weights.ndim != 1 or sensor_weights.shape[0] != n_sensors:
      raise ValueError(
          "Expected sensor_weights to be a numpy array of shape (n_sensors,), "
          f"received {sensor_weights.shape}"
      )
    return sensor_weights
  else:
    weights = np.ones(n_sensors)
    ids = get_sensor_indices(model, list(sensor_weights.keys()))
    for i, w in zip(ids, sensor_weights.values()):
      weights[i] = w
    return weights


def weighted_diff(
    predicted_data: np.ndarray,
    measured_data: np.ndarray,
    model: mujoco.MjModel | None = None,
    sensor_weights: Mapping[str, float] | np.ndarray | None = None,
) -> np.ndarray:
  """Compute the difference `measured_data - predicted_data`, optionally scaled by
  sensor weights.

  Args:
    predicted_data: The predicted data, of shape (n_timesteps, n_sensors).
    measured_data: The measured data, of shape (n_timesteps, n_sensors).
    sensor_weights: An optional dict mapping sensor name to weight. Unspecified sensors
      are assumed to have a weight of 1.
    model: Optional mujoco model. This argument is required if sensor_weights is not
      None.

  Returns:
    A numpy array of the weighted difference.
  """
  res = measured_data - predicted_data
  if sensor_weights is None:
    return res
  if model is None:
    raise ValueError("model is required if sensor_weights is provided")
  return res * prepare_sensor_weights(sensor_weights, res.shape[-1], model)


def normalize_residual(
    residual: np.ndarray,
    measured_data: np.ndarray,
) -> np.ndarray:
  """Normalize the residual by the standard deviation of the measured data."""
  return residual / (np.linalg.norm(measured_data, axis=0) / np.sqrt(2))
