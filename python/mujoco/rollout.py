# Copyright 2022 DeepMind Technologies Limited
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
"""Roll out open-loop trajectories from initial states, get subsequent states and sensor values."""

from typing import Optional, Union

import mujoco
from mujoco import _rollout
import numpy as np
from numpy import typing as npt


def rollout(model: Union[mujoco.MjModel, list[mujoco.MjModel]],
            data: Union[mujoco.MjData, list[mujoco.MjData]],
            initial_state: Optional[Union[npt.ArrayLike, list[npt.ArrayLike]]] = None,
            control: Optional[Union[npt.ArrayLike, list[npt.ArrayLike]]] = None,
            *,  # require subsequent arguments to be named
            control_spec: Union[int, list[int]] = mujoco.mjtState.mjSTATE_CTRL.value,
            skip_checks: bool = False,
            nroll: Optional[Union[int, list[int]]] = None,
            nstep: Optional[Union[int, list[int]]] = None,
            initial_warmstart: Optional[Union[npt.ArrayLike, list[npt.ArrayLike]]] = None,
            state: Optional[Union[npt.ArrayLike, list[npt.ArrayLike]]] = None,
            sensordata: Optional[Union[npt.ArrayLike, list[npt.ArrayLike]]] = None,
            nthread: Optional[int] = None):
  """Rolls out open-loop trajectories from initial states, get subsequent states and sensor values.

  Python wrapper for rollout.cc, see documentation therein.
  Infers nroll and nstep.
  Tiles inputs with singleton dimensions.
  Allocates outputs if none are given.

  Args: # TODO describe args better
    model: An mjModel or list of mjModel instances.
    data: Associated mjData instance(s).
    initial_state: Array of initial states from which to roll out trajectories.
      ([nroll or 1] x nstate)
    control: Open-loop controls array to apply during the rollouts.
      ([nroll or 1] x [nstep or 1] x ncontrol)
    control_spec: mjtState specification of control vectors.
    skip_checks: Whether to skip internal shape and type checks.
    nroll: Number of rollouts (inferred if unspecified).
    nstep: Number of steps in rollouts (inferred if unspecified).
    initial_warmstart: Initial qfrc_warmstart array (optional).
      ([nroll or 1] x nv)
    state: State output array (optional).
      (nroll x nstep x nstate)
    sensordata: Sensor data output array (optional).
      (nroll x nstep x nsensordata)

  Returns:
    state:
      State output array, (nroll x nstep x nstate).
    sensordata:
      Sensor data output array, (nroll x nstep x nsensordata).

  Raises:
    ValueError: bad shapes or sizes.
  """
  # skip_checks shortcut:
  #   don't infer nroll/nstep
  #   don't support singleton expansion
  #   don't allocate output arrays
  #   just call rollout and return
  if skip_checks:
    _rollout.rollout(model, data, nroll, nstep, control_spec, initial_state,
                     initial_warmstart, control, state, sensordata, nthread)
    return state, sensordata

  # check model, data for consistent length
  if isinstance(model, list):
    if not isinstance(data, list):
      raise ValueError('model and data must be single instances or lists')
    if len(model) != len(data):
      raise ValueError('model and data must be the same length')

  # check types
  if nthread and not isinstance(nthread, int):
    raise ValueError('nthread must be an integer')

  # convert args allowed to be single objects to lists
  model = _ensure_in_list(model)
  data = _ensure_in_list(data)
  if not initial_state:
    initial_state = []
    for m, d in zip(model, data):
      initial_state.append(np.zeros((mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS),)))
      mujoco.mj_getState(m, d, initial_state[-1], mujoco.mjtState.mjSTATE_FULLPHYSICS)
  else:
    initial_state = _ensure_in_list(initial_state)

  # record if optional args were unspecified
  control_none = control is None
  initial_warmstart_none = initial_warmstart is None
  state_none = state is None
  sensordata_none = sensordata is None

  # convert optional args allowed to be single objects to lists
  control_spec      = _ensure_in_list(control_spec, len(model))
  nroll             = _ensure_in_list(nroll, len(model))
  nstep             = _ensure_in_list(nstep, len(model))
  control           = _ensure_in_list(control, len(model))
  initial_warmstart = _ensure_in_list(initial_warmstart, len(model))
  state             = _ensure_in_list(state, len(model))
  sensordata        = _ensure_in_list(sensordata, len(model))

  for i in range(len(model)):
    # check control_spec
    if control_spec[i] & ~mujoco.mjtState.mjSTATE_USER.value:
      raise ValueError('control_spec can only contain bits in mjSTATE_USER')

    # check types
    if nroll[i] and not isinstance(nroll[i], int):
      raise ValueError('nroll must be an integer')
    if nstep[i] and not isinstance(nstep[i], int):
      raise ValueError('nstep must be an integer')

    _check_must_be_numeric(
        initial_state=initial_state[i],
        initial_warmstart=initial_warmstart[i],
        control=control[i],
        state=state[i],
        sensordata=sensordata[i])

    # check number of dimensions
    _check_number_of_dimensions(2,
                                initial_state=initial_state[i],
                                initial_warmstart=initial_warmstart[i])
    _check_number_of_dimensions(3,
                                control=control[i],
                                state=state[i],
                                sensordata=sensordata[i])

    # ensure 2D, make contiguous, row-major (C ordering)
    initial_state[i] = _ensure_2d(initial_state[i])
    initial_warmstart[i] = _ensure_2d(initial_warmstart[i])

    # ensure 3D, make contiguous, row-major (C ordering)
    control[i] = _ensure_3d(control[i])
    state[i] = _ensure_3d(state[i])
    sensordata[i] = _ensure_3d(sensordata[i])

    # check trailing dimensions
    nstate = mujoco.mj_stateSize(model[i], mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    _check_trailing_dimension(nstate, initial_state=initial_state[i], state=state[i])
    ncontrol = mujoco.mj_stateSize(model[i], control_spec[i])
    _check_trailing_dimension(ncontrol, control=control[i])
    _check_trailing_dimension(model[i].nv, initial_warmstart=initial_warmstart[i])
    _check_trailing_dimension(model[i].nsensordata, sensordata=sensordata[i])

    # infer nroll, check for incompatibilities
    nroll[i] = _infer_dimension(0, nroll[i] or 1,
                             initial_state=initial_state[i],
                             initial_warmstart=initial_warmstart[i],
                             control=control[i],
                             state=state[i],
                             sensordata=sensordata[i])

    # infer nstep, check for incompatibilities
    nstep[i] = _infer_dimension(1, nstep[i] or 1,
                                control=control[i],
                                state=state[i],
                                sensordata=sensordata[i])

    # tile input arrays if required (singleton expansion)
    initial_state[i] = _tile_if_required(initial_state[i], nroll[i])
    initial_warmstart[i] = _tile_if_required(initial_warmstart[i], nroll[i])
    control[i] = _tile_if_required(control[i], nroll[i], nstep[i])

    # allocate output if not provided
    if state[i] is None:
      state[i] = np.empty((nroll[i], nstep[i], nstate))
    if sensordata[i] is None:
      sensordata[i] = np.empty((nroll[i], nstep[i], model[i].nsensordata))

  control = _restore_none(control, control_none)
  initial_warmstart = _restore_none(initial_warmstart, initial_warmstart_none)
  state = _restore_none(state, state_none)
  sensordata = _restore_none(sensordata, sensordata_none)

  # call rollout
  _rollout.rollout(model, data, nroll, nstep, control_spec, initial_state,
                   initial_warmstart, control, state, sensordata, nthread)

  # return outputs
  return state, sensordata

def _check_must_be_numeric(**kwargs):
  for key, value in kwargs.items():
    if value is None:
      continue
    if not isinstance(value, np.ndarray) and not isinstance(value, float):
      raise ValueError(f'{key} must be a numpy array or float')


def _check_number_of_dimensions(ndim, **kwargs):
  for key, value in kwargs.items():
    if value is None:
      continue
    if value.ndim > ndim:
      raise ValueError(f'{key} can have at most {ndim} dimensions')


def _check_trailing_dimension(dim, **kwargs):
  for key, value in kwargs.items():
    if value is None:
      continue
    if value.shape[-1] != dim:
      raise ValueError(
          f'trailing dimension of {key} must be {dim}, got {value.shape[-1]}'
      )

def _ensure_in_list(arg, length=1):
  if arg is None:
    return [arg]*length
  elif type(arg) != list:
    return [arg]*length
  else:
    return arg

def _restore_none(arg, is_none):
  return None if is_none else arg

def _ensure_2d(arg):
  if arg is None:
    return None
  else:
    return np.ascontiguousarray(np.atleast_2d(arg), dtype=np.float64)


def _ensure_3d(arg):
  if arg is None:
    return None
  else:
    # np.atleast_3d adds both leading and trailing dims, we want only leading
    if arg.ndim == 0:
      arg = arg[np.newaxis, np.newaxis, np.newaxis, ...]
    elif arg.ndim == 1:
      arg = arg[np.newaxis, np.newaxis, ...]
    elif arg.ndim == 2:
      arg = arg[np.newaxis, ...]
    return np.ascontiguousarray(arg, dtype=np.float64)


def _infer_dimension(dim, value, **kwargs):
  """Infers dimension `dim` given guess `value` from set of arrays.

  Args:
    dim: Dimension to be inferred.
    value: Initial guess of inferred value (1: unknown).
    **kwargs: List of arrays which should all have the same size (or 1)
      along dimension dim.

  Returns:
    Inferred dimension.

  Raises:
    ValueError: If mismatch between array shapes or initial guess.
  """
  for name, array in kwargs.items():
    if array is None:
      continue
    if array.shape[dim] != value:
      if value == 1:
        value = array.shape[dim]
      elif array.shape[dim] != 1:
        raise ValueError(
            f'dimension {dim} inferred as {value} '
            f'but {name} has {array.shape[dim]}'
        )
  return value


def _tile_if_required(array, dim0, dim1=None):
  if array is None:
    return
  reps = np.ones(array.ndim, dtype=int)
  if array.shape[0] == 1:
    reps[0] = dim0
  if dim1 is not None and array.shape[1] == 1:
    reps[1] = dim1
  return np.tile(array, reps)
