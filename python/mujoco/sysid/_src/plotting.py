# Copyright 2026 DeepMind Technologies Limited
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

"""Plotting utilities."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import mujoco
import numpy as np


def plot_sensor_comparison(
    model: mujoco.MjModel,
    predicted_times: np.ndarray | None = None,
    predicted_data: np.ndarray | None = None,
    real_data: np.ndarray | None = None,
    real_times: np.ndarray | None = None,
    preid_data: np.ndarray | None = None,
    preid_times: np.ndarray | None = None,
    commanded_data: np.ndarray | None = None,
    commanded_times: np.ndarray | None = None,
    size_factor: float = 1.0,
    title_prefix: str = "",
    sensor_ids: list[int] | None = None,
):
  """Plots sensor trajectories from simulation and real data.

  Args:
      model: The model object providing sensor information.
      predicted_times: Optional 1D array of timestamps corresponding to
        simulation data.
      predicted_data: Optional 2D array of simulation sensor data with shape
        (num_timesteps, sensor_data_dimension).
      real_data: Optional 2D array of real sensor data with the same shape as
        predicted_data.
      real_times: A 1D array of timestamps corresponding to real data. If None
        and real_data is provided, the first available timestamp array is used.
      preid_data: Optional 2D array of pre-identification sensor data.
      preid_times: A 1D array of timestamps for pre-identification data.
      commanded_data: Optional 2D array of commanded sensor data.
      commanded_times: A 1D array of timestamps for commanded data.
      size_factor: A scaling factor for the figure size.
      title_prefix: Optional prefix for subplot titles.
      sensor_ids: Optional list of sensor indices to plot.
  """
  # Define a more appealing color palette
  predicted_color = "#1f77b4"  # Steel blue
  real_color = "#ff7f0e"  # Safety orange
  preid_color = "#2ca02c"  # Forest green
  commanded_color = "#9467bd"  # Purple

  # Determine the reference time array to use
  reference_times = None
  if predicted_times is not None:
    reference_times = predicted_times
  elif real_times is not None:
    reference_times = real_times
  elif preid_times is not None:
    reference_times = preid_times
  elif commanded_times is not None:
    reference_times = commanded_times
  else:
    raise ValueError("At least one time array must be provided")

  # Set times for data sources that don't have their own time arrays
  if real_data is not None and real_times is None:
    real_times = reference_times
  if preid_data is not None and preid_times is None:
    preid_times = reference_times
  if commanded_data is not None and commanded_times is None:
    commanded_times = reference_times
  if predicted_data is not None and predicted_times is None:
    predicted_times = reference_times

  if sensor_ids is None:
    sensor_ids = list(range(model.nsensor))
  assert predicted_data is not None
  n_plots = predicted_data.shape[1]

  fig, axes = plt.subplots(
      n_plots,
      1,
      figsize=(10 * size_factor, 2.5 * n_plots * size_factor),
      sharex=True,
  )
  if n_plots == 1:
    axes = [axes]
  axes = list(axes)  # pyright: ignore[reportArgumentType]

  # Set an overall title for the figure.
  fig.suptitle(title_prefix + " Sensors", fontsize=14)  # , y=1.02)

  # Loop over each sensor.
  plot_i = 0
  sensor_dim = 1
  j = 0
  dim_str = ""
  for sensor_id in sensor_ids:
    sensor = model.sensor(sensor_id)
    sensor_name = sensor.name
    sensor_dim = int(sensor.dim[0])
    sensor_addr = int(sensor.adr[0])

    for j in range(sensor_dim):
      ax = axes[plot_i]
      plot_i += 1
      dim_str = "" if sensor_dim == 1 else f" {j}"
      if predicted_data is not None:
        assert predicted_times is not None
        predicted_signal = predicted_data[
            :, sensor_addr : sensor_addr + sensor_dim
        ]
        ax.plot(
            predicted_times,
            predicted_signal[:, j],
            lw=2,
            color=predicted_color,
            alpha=0.8,
            label="Sim" + dim_str,
        )
      if real_data is not None:
        assert real_times is not None
        real_signal = real_data[:, sensor_addr : sensor_addr + sensor_dim]
        ax.plot(
            real_times,
            real_signal[:, j],
            lw=2,
            color=real_color,
            linestyle="--",
            alpha=0.7,
            label="Real" + dim_str,
        )
      if preid_data is not None:
        assert preid_times is not None
        preid_signal = preid_data[:, sensor_addr : sensor_addr + sensor_dim]
        ax.plot(
            preid_times,
            preid_signal[:, j],
            lw=2,
            color=preid_color,
            linestyle=":",
            alpha=0.6,
            label="Pre-ID" + dim_str,
        )
      if commanded_data is not None:
        assert commanded_times is not None
        commanded_signal = commanded_data[
            :, sensor_addr : sensor_addr + sensor_dim
        ]
        ax.plot(
            commanded_times,
            commanded_signal[:, j],
            lw=2,
            color=commanded_color,
            linestyle="-.",
            alpha=0.6,
            label="Commanded" + dim_str,
        )
      # Place the sensor name in a white box in the top-left corner.
      ax.text(
          0.02,
          0.9,
          sensor_name + dim_str,
          transform=ax.transAxes,
          fontsize=10,
          weight="bold",
          verticalalignment="top",
          horizontalalignment="left",
          bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
      )

      # Enable a dashed grid.
      ax.grid(True, linestyle="--", alpha=0.7)

  # Loop over "extra" sensors from the user
  for _ in range(plot_i, n_plots):
    sensor_name = "user_sensor"
    dim_str = "" if sensor_dim == 1 else f" {j}"
    ax = axes[plot_i]
    plot_i += 1
    if predicted_data is not None:
      assert predicted_times is not None
      predicted_signal = predicted_data[:, plot_i - 1]
      ax.plot(
          predicted_times,
          predicted_signal,
          lw=2,
          color=predicted_color,
          alpha=0.8,
          label="Sim",
      )
    if real_data is not None:
      assert real_times is not None
      real_signal = real_data[:, plot_i - 1]
      ax.plot(
          real_times,
          real_signal,
          lw=2,
          color=real_color,
          linestyle="--",
          alpha=0.7,
          label="Real",
      )
    if preid_data is not None:
      assert preid_times is not None
      preid_signal = preid_data[:, plot_i - 1]
      ax.plot(
          preid_times,
          preid_signal,
          lw=2,
          color=preid_color,
          linestyle=":",
          alpha=0.6,
          label="Pre-ID",
      )
    if commanded_data is not None:
      assert commanded_times is not None
      commanded_signal = commanded_data[:, plot_i - 1]
      ax.plot(
          commanded_times,
          commanded_signal,
          lw=2,
          color=commanded_color,
          linestyle="-.",
          alpha=0.6,
          label="Commanded",
      )
    # Place the sensor name in a white box in the top-left corner.
    ax.text(
        0.02,
        0.9,
        sensor_name + dim_str,
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Enable a dashed grid.
    ax.grid(True, linestyle="--", alpha=0.7)

  # Add a unified, figure-level legend if any data is provided.
  legend_handles = []
  if predicted_data is not None:
    legend_handles.append(
        Line2D([0], [0], color=predicted_color, lw=2, label="Simulation")
    )
  if real_data is not None:
    legend_handles.append(
        Line2D([0], [0], color=real_color, lw=2, linestyle="--", label="Real")
    )
  if preid_data is not None:
    legend_handles.append(
        Line2D([0], [0], color=preid_color, lw=2, linestyle=":", label="Pre-ID")
    )
  if commanded_data is not None:
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=commanded_color,
            lw=2,
            linestyle="-.",
            label="Commanded",
        )
    )

  if legend_handles:
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(legend_handles),
        fancybox=True,
        shadow=True,
        fontsize=10,
        title="Data Source",
    )

  fig.supxlabel("Time (s)", fontsize=8)
  plt.tight_layout(rect=(0, 0.03, 1, 0.9))


def render_rollout(
    model: mujoco.MjModel | Sequence[mujoco.MjModel],
    data: mujoco.MjData,
    state: np.ndarray,
    framerate: int,
    camera: str | int = -1,
    width: int = 640,
    height: int = 480,
    light_pos: Sequence[float] | None = None,
) -> list[np.ndarray]:
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

  if isinstance(model, mujoco.MjModel):
    models_list = [model] * nbatch
  else:
    models_list = list(model)
    if len(models_list) == 1:
      models_list = models_list * nbatch
    else:
      assert len(models_list) == nbatch

  # Visual options
  vopt = mujoco.MjvOption()
  vopt.geomgroup[3] = 1  # Show visualization geoms

  pert = mujoco.MjvPerturb()
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

  # Simulate and render.
  frames = []

  with mujoco.Renderer(models_list[0], height=height, width=width) as renderer:
    for i in range(state.shape[1]):
      # Check if we should capture this frame based on framerate
      if len(frames) < i * models_list[0].opt.timestep * framerate:
        for j in range(state.shape[0]):
          # Set state
          mujoco.mj_setState(
              models_list[j],
              data,
              state[j, i, :],
              mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
          )
          mujoco.mj_forward(models_list[j], data)

          # Use first model to make the scene, add subsequent models
          if j == 0:
            renderer.update_scene(data, camera, scene_option=vopt)
          else:
            mujoco.mjv_addGeoms(
                models_list[j], data, vopt, pert, catmask, renderer.scene
            )

        # Add light, if requested
        if light_pos is not None:
          if renderer.scene.nlight < 100:  # check limit
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
