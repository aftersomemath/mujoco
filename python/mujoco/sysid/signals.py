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
"""Signal generation functions."""

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np
import scipy.signal as sp

from mujoco.sysid import timeseries


def chirp_wave(
    t: np.ndarray,
    f0: float,
    f1: float,
    duration: float,
    tau: float = 0.0,
    method: str = 'linear',
    phase_shift: float = 0.0,
) -> np.ndarray:
  """Chirp signal with amplitude decay.

  Args:
    t: Time array for signal generation.
    f0: Initial frequency in Hz.
    f1: Final frequency in Hz.
    duration: Total duration of the signal in seconds.
    tau: Exponential decay time constant. If > 0, applies exponential decay
      to the signal amplitude. Default is 0.0 (no decay).
    method: Kind of frequency sweep. If not given, linear is assumed.
    phase_shift: Phase shift in degrees. Defaults to 0.

  Returns:
    np.ndarray: Chirp signal values corresponding to the input time array.
  """
  t_norm = t - t[0]
  initial_phase = -90 + phase_shift
  signal = sp.chirp(
      t_norm, f0=f0, f1=f1, t1=duration, method=method, phi=initial_phase
  )
  if tau > 0:
    envelope = np.exp(-t_norm / duration / tau)
    return signal * envelope
  return signal


def get_ctrl(
    model: mujoco.MjModel,
    f0: float,
    f1: float,
    duration: float,
    dt: float | None = None,
    tau: float = 0.0,
    df1: float = 1.0,
    seed: int = 0,
    initial_qpos: np.ndarray | None = None,
    ctrlrange_scale: float | Sequence[float] = 1.0,
    symmetry: Sequence[str] | None = None,
    method: str = 'linear',
    amplitude_scale: float = 1.0,
    phase_shift: float | Sequence[float] = 0.0,
) -> timeseries.TimeSeries:
  """Generate control signals for MuJoCo actuators using chirp waves.

  Args:
    model: MuJoCo model object containing actuator information.
    f0: Initial frequency in Hz for all actuators.
    f1: Base final frequency in Hz (will be scaled per actuator).
    duration: Total duration of the signal in seconds.
    dt: Time step for signal generation. If None, uses model's timestep.
    tau: Exponential decay time constant for signal amplitude.
    df1: Factor for randomizing final frequencies across actuators.
      Final frequencies will be uniformly sampled between f1/df1 and f1*df1.
    seed: Random seed for reproducible frequency scaling.
    initial_qpos: Optional initial position command for each actuator. If
      provided, the generated signal will be offset so that its first value
      matches this initial position.
    ctrlrange_scale: Scale factor for the control range.
    method: Kind of frequency sweep. If not given, logarithmic is assumed.
    amplitude_scale: Overall scale factor for the signal amplitude variation.
    phase_shift: Phase shift in degrees for the chirp signal. Can be a scalar
      (applied to all actuators) or a sequence of length model.nu.
  Returns:
    TimeSeries: Control signals as a TimeSeries instance.

  Raises:
    ValueError: If dt is not positive.
    ValueError: If `ctrlrange_scale` or `initial_qpos` dimensions mismatch `model.nu`.

  The function works as follows:
  1. A base chirp signal `y` (ranging from -1 to 1) is generated for each actuator.
  2. The hardware control limits (`lower_hw`, `upper_hw`) are retrieved.
  3. The desired signal amplitude `A` for each actuator is calculated as
     `A = (upper_hw - lower_hw) / 2 * ctrlrange_scale`.
     This `A` represents the target amplitude of the signal variation for each actuator,
     scaled based on the hardware limits and the requested `ctrlrange_scale`.
  4. If `initial_qpos` is provided, the signal is designed to start exactly at the initial
     position and oscillate around that value.
     - The signal variation `v = (y - y[0]) * A * amplitude_scale` is calculated.
     - The target signal is `target = initial_qpos + v`.
  5. If `initial_qpos` is not provided, the signal is centered around the midpoint of the
      allowed range.
     - The center `c = (lower_hw + upper_hw) / 2` is calculated.
     - The signal variation `v = y * A * amplitude_scale` is calculated relative to the center.
     - The target signal is `target = c + v`.
  6. Actuators with `ctrlrange_scale == 0` have their signal set to `initial_qpos`
     (if provided) or `c`.
  7. The final signal is clipped to the hardware limits `[lower_hw, upper_hw]`.
  """
  if dt is None:
    dt = model.opt.timestep
  if dt <= 0:
    raise ValueError('dt must be a positive float.')

  # Create the time vector with uniform steps exactly equal to dt.
  # Using np.arange ensures that each step is exactly dt.
  x = np.arange(0, duration + dt / 10, dt)

  # Sampling scaling factors in [1/df1, df1] so that each final frequency is:
  # f1_i = f1 * s, where s in [1/df1, df1]
  rng = np.random.default_rng(seed)
  scales = rng.uniform(1 / df1, df1, size=model.nu)
  f1_vector = f1 * scales
  print(f'f1_vector: {f1_vector}')

  # Handle phase_shift input (scalar or sequence)
  phase_shift_array = np.asarray(phase_shift)
  if phase_shift_array.ndim == 0:
    phase_shift_array = np.full(model.nu, phase_shift_array.item())
  elif phase_shift_array.ndim == 1:
    if len(phase_shift_array) != model.nu:
      raise ValueError(
          f'Length of phase_shift ({len(phase_shift_array)}) must match '
          f'number of actuators ({model.nu})'
      )
  else:
    raise ValueError('phase_shift must be a scalar or a 1D array.')
  print(f'phase_shift_array: {phase_shift_array}')

  # Generate chirp signals for each actuator and stack them column-wise.
  y = np.vstack([
      chirp_wave(
          x,
          f0=f0,
          duration=duration,
          f1=f1_i,
          tau=tau,
          method=method,
          phase_shift=ps,
      )
      for f1_i, ps in zip(f1_vector, phase_shift_array)
  ]).T

  if symmetry is not None:
    y[:, symmetry] *= -1

  scale_array = np.asarray(ctrlrange_scale)
  if scale_array.ndim == 0:
    scale_array = np.ones(model.nu) * scale_array
  elif scale_array.ndim == 1:
    if len(scale_array) != model.nu:
      raise ValueError(
          f'Length of ctrlrange_scale ({len(scale_array)}) must match '
          f'number of actuators ({model.nu})'
      )

  # Get hardware control limits and calculate center and base amplitude
  lower_hw, upper_hw = model.actuator_ctrlrange.T
  center = (lower_hw + upper_hw) / 2.0
  amplitude = (upper_hw - lower_hw) / 2.0 * scale_array

  y_final = np.zeros_like(y)
  active_actuators = scale_array != 0
  inactive_actuators = scale_array == 0

  # Calculate signal for active actuators
  if initial_qpos is not None:
    if len(initial_qpos) != model.nu:
      raise ValueError(
          f'Length of initial_qpos ({len(initial_qpos)}) must match '
          f'number of actuators ({model.nu})'
      )
    # Calculate the scaled variation directly from the base chirp y (-1 to 1),
    # scale it by the desired amplitude (amplitude * amplitude_scale),
    # center this oscillation around initial_qpos, then clip to hw limits.
    y_variation_scaled = (
        y[:, active_actuators] * amplitude[active_actuators] * amplitude_scale
    )
    y_target = initial_qpos[active_actuators] + y_variation_scaled
    y_final[:, active_actuators] = np.clip(
        y_target, lower_hw[active_actuators], upper_hw[active_actuators]
    )
  else:
    # Center the base signal (-1 to 1) within the scaled range
    y_centered = (
        center[active_actuators]
        + y[:, active_actuators] * amplitude[active_actuators]
    )
    # Apply amplitude_scale to the variation around the center
    y_variation = (y_centered - center[active_actuators]) * amplitude_scale
    y_final[:, active_actuators] = center[active_actuators] + y_variation
    # Clip to hardware limits (important if amplitude_scale > 1 or scale_array > 1)
    y_final[:, active_actuators] = np.clip(
        y_final[:, active_actuators],
        lower_hw[active_actuators],
        upper_hw[active_actuators],
    )

  # Set signal for inactive actuators
  if np.any(inactive_actuators):
    if initial_qpos is not None:
      # Use initial_qpos if provided
      y_final[:, inactive_actuators] = initial_qpos[inactive_actuators]
    else:
      # Otherwise, use the center of the hardware range
      y_final[:, inactive_actuators] = center[inactive_actuators]

  # Clip the final signal for inactive actuators to hardware limits as well
  y_final[:, inactive_actuators] = np.clip(
      y_final[:, inactive_actuators],
      lower_hw[inactive_actuators],
      upper_hw[inactive_actuators],
  )

  return timeseries.TimeSeries(x, y_final)


def scale_amplitude(
    ts: timeseries.TimeSeries, factor: float | np.ndarray
) -> timeseries.TimeSeries:
  """Scale the amplitude of the signal.

  Args:
    ts: TimeSeries to scale.
    factor: Scalar or 1D array of scaling factors.

  Returns:
    TimeSeries: Scaled signal.

  Raises:
    ValueError: If `factor` is not a scalar or a 1D array.
    ValueError: If `factor` has the wrong number of elements.
  """
  scale = np.atleast_1d(factor)
  if scale.ndim == 0:
    scale = np.ones(ts.data.shape[1]) * scale
  elif scale.ndim == 1:
    if len(scale) != ts.data.shape[1]:
      raise ValueError(
          f'Length of scale ({len(scale)}) must match '
          f'number of actuators ({ts.data.shape[1]})'
      )
  else:
    raise ValueError('scale must be a scalar or a 1D array.')
  print(f'Scaling signal by {scale}')
  return timeseries.TimeSeries(ts.times, ts.data * scale)


def scale_time(
    ts: timeseries.TimeSeries, factor: float
) -> timeseries.TimeSeries:
  new_times = np.linspace(ts.times[0], ts.times[-1] / factor, num=len(ts.times))
  new_data = np.empty_like(ts.data)
  for i in range(ts.data.shape[1]):
    new_data[:, i] = np.interp(factor * new_times, ts.times, ts.data[:, i])
  return timeseries.TimeSeries(new_times, new_data)


def _brownian_noise(
    size: tuple,
    dt: float,
    tau: float,
    std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
  # If tau is 0, return white noise.
  if tau == 0:
    return rng.normal(scale=std, size=size)

  # Compute the scale such that the stationary variance is std**2.
  rate = np.exp(-dt / tau)  # 0 < rate < 1
  std = np.atleast_1d(std)
  scale = std * np.sqrt(1 - rate**2)

  white_noise = rng.normal(size=size)
  noise = np.zeros(size)
  for j in range(size[1]):
    # noise[t] = rate * noise[t-1] + scale[j]*white_noise[t].
    noise[:, j] = sp.lfilter([scale[j]], [1, -rate], white_noise[:, j])
  return noise


def add_ornstein_uhlenbeck_noise(
    sensordata: np.ndarray,
    dt: float,
    tau: float,
    std: float | np.ndarray,
    seed: int,
) -> np.ndarray:
  """Adds Ornstein-Uhlenbeck noise to the sensor data.

  Args:
    sensordata: Sensor data to add noise to. Shape (n_steps, n_sensors).
    dt: Time step duration.
    tau: Time constant controlling the temporal correlation. If tau = 0,
      uncorrelated noise is generated. As tau increases, the noise becomes more
      correlated.
    std: Standard deviation of the noise. Either a scalar or a 1D array of
      length n_sensors.
    seed: Random seed for reproducibility.

  Returns:
    np.ndarray: Sensor data with Ornstein-Uhlenbeck noise added.
  """
  if tau < 0:
    raise ValueError(f'tau must be non-negative, got {tau}')
  rng = np.random.default_rng(seed)

  if np.isscalar(std):
    std_array = np.full(sensordata.shape[-1], std)
  else:
    std_array = np.asarray(std)

  if std_array.shape[0] != sensordata.shape[-1]:
    raise ValueError(
        'The length of std must match the number of sensor channels.'
        f'Got {std_array.shape[0]} and {sensordata.shape[-1]}.'
    )
  return sensordata + _brownian_noise(sensordata.shape, dt, tau, std_array, rng)


# def ellipsoidal_cartesian_trajectory(
#     duration: float,
#     dt: float,
#     ellipsoid_radii: Tuple[float, float, float] = (1.0, 1.0, 1.0),
#     A_phi: float = 12.0,
#     A_theta: float = 3.4,
#     omega_phi: float = 1.63,
#     omega_theta: float = 0.265,
#     initial_phi: float = 0.0,
#     initial_theta: float = 0.0,
#     origin: Optional[Tuple[float, float, float]] = None,
# ) -> timeseries.TimeSeries:
#   """Generates a Cartesian trajectory on an ellipsoidal shell.

#   The target point on the shell is defined using spherical angles (phi, theta),
#   where phi is the azimuthal angle (in the xy-plane) and theta is the polar
#   angle (from the z-axis). The rates of change for these angles are sinusoidal
#   functions of time:

#   phi_dot = A_phi * sin(omega_phi * t)
#   theta_dot = A_theta * sin(omega_theta * t)

#   Args:
#     duration: Total duration of the trajectory in seconds.
#     dt: Time step for signal generation (sampling interval).
#     ellipsoid_radii: Tuple (a, b, c) representing the radii of the ellipsoid along the
#       x, y, and z axes respectively. Defaults to a unit sphere.
#     A_phi: Amplitude of the azimuthal angle rate (rad/s).
#     A_theta: Amplitude of the polar angle rate (rad/s).
#     omega_phi: Frequency of the azimuthal angle rate (rad/s).
#     omega_theta: Frequency of the polar angle rate (rad/s).
#     initial_phi: Initial azimuthal angle (radians). Defaults to 0.
#     initial_theta: Initial polar angle (radians). Defaults to 0.
#     origin: Tuple (x, y, z) representing the origin of the ellipsoid. Defaults to
#       (0, 0, 0).

#   Returns:
#       A TimeSeries object containing the time vector and the (x, y, z) coordinates of
#       the target point.

#   Raises:
#       ValueError: If dt is not positive or angular frequencies are zero.
#   """
#   if dt <= 0:
#     raise ValueError("dt must be a positive float.")
#   if omega_phi == 0 or omega_theta == 0:
#     raise ValueError(
#         "Angular frequencies (omega_phi, omega_theta) cannot be zero."
#     )

#   t = np.arange(0, duration + dt / 10, dt)

#   # Calculate angles phi and theta over time using analytical integration:
#   # phi(t) = phi(0) + integral(A_phi * sin(omega_phi * tau) d tau) from 0 to t
#   # theta(t) = theta(0) + integral(A_theta * sin(omega_theta * tau) d tau) from 0 to t
#   phi = initial_phi + (A_phi / omega_phi) * (1 - np.cos(omega_phi * t))
#   theta = initial_theta + (A_theta / omega_theta) * (
#       1 - np.cos(omega_theta * t)
#   )

#   # Convert spherical coordinates to Cartesian coordinates.
#   a, b, c = ellipsoid_radii
#   x = a * np.sin(theta) * np.cos(phi)
#   y = b * np.sin(theta) * np.sin(phi)
#   z = c * np.cos(theta)

#   xyz = np.vstack([x, y, z]).T

#   # Shift to start at the origin.
#   if origin is not None:
#     origin = np.asarray(origin)
#     shift = origin - xyz[0]
#     xyz = xyz + shift

#   return sensordata + _brownian_noise(sensordata.shape, dt, tau, std_array, rng)
