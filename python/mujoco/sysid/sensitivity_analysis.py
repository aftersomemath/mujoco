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
"""Sensitivity analysis utilities."""

from typing import Callable

import numpy as np

from mujoco_sysid import parameter


def sweep_parameter(
    param: parameter.ParameterDict,
    param_name: str,
    values: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
  """Sweep a parameter over a range of values and return the cost."""
  costs = []
  for val in values:
    param.reset()
    param[param_name].update_from_vector(val)
    x = param.as_vector()
    res = residual_fn(x)
    cost = np.sum(res**2)
    costs.append(cost)
  return np.array(costs)
