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
"""Tests for residual."""

import mujoco
import numpy as np

from mujoco.sysid._src import residual
from mujoco.sysid._src import timeseries


RESIDUAL_TEST_XML = """\
<mujoco>
  <worldbody>
    <body name="body" pos="0 0 0">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/>
      <site name="s1"/>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="j1_pos" joint="j1"/>
    <accelerometer name="acc1" site="s1"/>
  </sensor>
</mujoco>
"""


def test_construct_ts_from_defaults():
  """Test mapping of sensor and state data to predicted timeseries."""
  model = mujoco.MjModel.from_xml_string(RESIDUAL_TEST_XML)
  n_steps = 5
  times = np.linspace(0, 1, n_steps)

  # pred_sensordata - contains j1_pos and acc1
  # j1_pos is sensor 0, acc1 is sensor 1,2,3
  pred_sensor_data = np.zeros((n_steps, 4))
  pred_sensor_data[:, 0] = np.arange(n_steps)
  pred_sensor_data[:, 1:4] = 10 * np.arange(n_steps * 3).reshape((n_steps, 3))
  pred_sensordata = timeseries.TimeSeries.from_names(
      times, pred_sensor_data, model, names=None
  )

  # state_ts - contains j1_qpos and j1_qvel
  qpos_map, qvel_map, act_map, _ = (
      timeseries.TimeSeries.compute_all_state_mappings(model)
  )
  state_map = qpos_map | qvel_map | act_map
  state_data = np.zeros((n_steps, model.nq + model.nv + model.na))
  state_data[:, state_map["j1_qpos"][1]] = (100 + np.arange(n_steps)).reshape(
      -1, 1
  )
  state_data[:, state_map["j1_qvel"][1]] = (200 + np.arange(n_steps)).reshape(
      -1, 1
  )
  state_ts = timeseries.TimeSeries(times, state_data, state_map)

  # measured_sensordata - defines target layout: j1_pos, j1_qpos, acc1
  measured_data = np.zeros((n_steps, 5))  # 1 + 1 + 3 = 5
  measured_signal_mapping = {
      "j1_pos": (timeseries.SignalType.MjSensor, np.array([0])),
      "j1_qpos": (timeseries.SignalType.MjStateQPos, np.array([1])),
      "acc1": (timeseries.SignalType.MjSensor, np.array([2, 3, 4])),
  }
  measured_sensordata = timeseries.TimeSeries(
      times, measured_data, measured_signal_mapping
  )

  # Test 1: enabled_observations=() - should use all from measured
  selected_measured, predicted_ts = residual.construct_ts_from_defaults(
      state_ts, pred_sensordata, measured_sensordata, enabled_observations=()
  )

  assert list(predicted_ts.signal_mapping.keys()) == ["j1_pos", "j1_qpos", "acc1"]
  # j1_pos = pred_sensordata[:,0] -> predicted_ts[:,0]
  np.testing.assert_allclose(predicted_ts.data[:, 0], np.arange(n_steps))
  # j1_qpos = state_ts -> predicted_ts[:,1]
  np.testing.assert_allclose(predicted_ts.data[:, 1], 100 + np.arange(n_steps))
  # acc1 = pred_sensordata[:,1:4] -> predicted_ts[:,2:5]
  np.testing.assert_allclose(
      predicted_ts.data[:, 2:5],
      10 * np.arange(n_steps * 3).reshape((n_steps, 3)),
  )

  # Test 2: enabled_observations subset
  enabled_subset = [
      ("j1_qpos", timeseries.SignalType.MjStateQPos),
      ("acc1", timeseries.SignalType.MjSensor),
  ]
  selected_measured2, predicted_ts2 = residual.construct_ts_from_defaults(
      state_ts,
      pred_sensordata,
      measured_sensordata,
      enabled_observations=enabled_subset,
  )

  assert list(predicted_ts2.signal_mapping.keys()) == ["j1_qpos", "acc1"]
  assert predicted_ts2.data.shape == (n_steps, 4)
  np.testing.assert_allclose(predicted_ts2.data[:, 0], 100 + np.arange(n_steps))
  np.testing.assert_allclose(
      predicted_ts2.data[:, 1:4],
      10 * np.arange(n_steps * 3).reshape((n_steps, 3)),
  )

  # Test 3: Reordered measured_sensordata: acc1, j1_qpos, j1_pos
  measured_signal_mapping_reordered = {
      "acc1": (timeseries.SignalType.MjSensor, np.array([0, 1, 2])),
      "j1_qpos": (timeseries.SignalType.MjStateQPos, np.array([3])),
      "j1_pos": (timeseries.SignalType.MjSensor, np.array([4])),
  }
  measured_sensordata_reordered = timeseries.TimeSeries(
      times, measured_data, measured_signal_mapping_reordered
  )
  selected_measured3, predicted_ts3 = residual.construct_ts_from_defaults(
      state_ts,
      pred_sensordata,
      measured_sensordata_reordered,
      enabled_observations=(),
  )
  assert list(predicted_ts3.signal_mapping.keys()) == [
      "acc1",
      "j1_qpos",
      "j1_pos",
  ]
  # acc1 = pred_sensordata[:,1:4] -> predicted_ts3[:,0:3]
  np.testing.assert_allclose(
      predicted_ts3.data[:, 0:3],
      10 * np.arange(n_steps * 3).reshape((n_steps, 3)),
  )
  # j1_qpos = state_ts -> predicted_ts3[:,3]
  np.testing.assert_allclose(
      predicted_ts3.data[:, 3], 100 + np.arange(n_steps)
  )
  # j1_pos = pred_sensordata[:,0] -> predicted_ts3[:,4]
  np.testing.assert_allclose(predicted_ts3.data[:, 4], np.arange(n_steps))

