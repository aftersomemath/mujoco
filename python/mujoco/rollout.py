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

from collections.abc import Sequence
from typing import Optional, Union

import mujoco
from mujoco import _rollout
import numpy as np
from numpy import typing as npt


def rollout(
    model: Union[mujoco.MjModel, Sequence[mujoco.MjModel]],
    data: Union[mujoco.MjData, Sequence[mujoco.MjData]],
    initial_state: npt.ArrayLike,
    control: Optional[npt.ArrayLike] = None,
    *,  # require subsequent arguments to be named
    control_spec: int = mujoco.mjtState.mjSTATE_CTRL.value,
    skip_checks: bool = False,
    nstep: Optional[int] = None,
    initial_warmstart: Optional[npt.ArrayLike] = None,
    state: Optional[npt.ArrayLike] = None,
    sensordata: Optional[npt.ArrayLike] = None,
    chunk_size: int = None,
):
  """Rolls out open-loop trajectories from initial states, get subsequent states and sensor values.

  Python wrapper for rollout.cc, see documentation therein.
  Infers nroll and nstep.
  Tiles inputs with singleton dimensions.
  Allocates outputs if none are given.

  Args:
    model: An instance or length nroll sequence of MjModel with the same size signature.
    data: Associated mjData instance or sequence of instances with length nthread.
    initial_state: Array of initial states from which to roll out trajectories.
      ([nroll or 1] x nstate)
    control: Open-loop controls array to apply during the rollouts.
      ([nroll or 1] x [nstep or 1] x ncontrol)
    control_spec: mjtState specification of control vectors.
    skip_checks: Whether to skip internal shape and type checks.
    nstep: Number of steps in rollouts (inferred if unspecified).
    initial_warmstart: Initial qfrc_warmstart array (optional).
      ([nroll or 1] x nv)
    state: State output array (optional).
      (nroll x nstep x nstate)
    sensordata: Sensor data output array (optional).
      (nroll x nstep x nsensordata)
    chunk_size: Determines threadpool chunk size. If unspecified,
                chunk_size = max(1, nroll / (nthread * 10)

  Returns:
    state:
      State output array, (nroll x nstep x nstate).
    sensordata:
      Sensor data output array, (nroll x nstep x nsensordata).

  Raises:
    ValueError: bad shapes or sizes.
  """  # fmt: skip
  # skip_checks shortcut:
  #   don't infer nroll/nstep
  #   don't support singleton expansion
  #   don't allocate output arrays
  #   just call rollout and return
  if skip_checks:
    _rollout.rollout(
        model,
        data,
        nstep,
        control_spec,
        initial_state,
        initial_warmstart,
        control,
        state,
        sensordata,
        chunk_size,
    )
    return state, sensordata

  if not isinstance(model, mujoco.MjModel):
    model = list(model)

  # check control_spec
  if control_spec & ~mujoco.mjtState.mjSTATE_USER.value:
    raise ValueError('control_spec can only contain bits in mjSTATE_USER')

  # check types
  if nstep and not isinstance(nstep, int):
    raise ValueError('nstep must be an integer')
  if chunk_size and not isinstance(chunk_size, int):
    raise ValueError('chunk_size must be an integer')
  _check_must_be_numeric(
      initial_state=initial_state,
      initial_warmstart=initial_warmstart,
      control=control,
      state=state,
      sensordata=sensordata,
  )

  # check number of dimensions
  _check_number_of_dimensions(
      2, initial_state=initial_state, initial_warmstart=initial_warmstart
  )
  _check_number_of_dimensions(
      3, control=control, state=state, sensordata=sensordata
  )

  # ensure 2D, make contiguous, row-major (C ordering)
  initial_state = _ensure_2d(initial_state)
  initial_warmstart = _ensure_2d(initial_warmstart)

  # ensure 3D, make contiguous, row-major (C ordering)
  control = _ensure_3d(control)
  state = _ensure_3d(state)
  sensordata = _ensure_3d(sensordata)

  # infer nroll, check for incompatibilities
  nroll = _infer_dimension(
      0,
      1,
      initial_state=initial_state,
      initial_warmstart=initial_warmstart,
      control=control,
      state=state,
      sensordata=sensordata,
  )
  if isinstance(model, list) and nroll == 1:
    nroll = len(model)

  if isinstance(model, list) and len(model) != nroll:
    raise ValueError(
        f'nroll inferred as {nroll} but model is length {len(model)}'
    )
  elif not isinstance(model, list):
    model = [model]  # Use a length 1 list to simplify code below

  if not isinstance(data, list):
    data = [data]  # Use a length 1 list to simplify code below

  # infer nstep, check for incompatibilities
  nstep = _infer_dimension(
      1, nstep or 1, control=control, state=state, sensordata=sensordata
  )

  # get nstate/ncontrol/nv/nsensordata
  # check that they are equal across models
  nstate = mujoco.mj_stateSize(
      model[0], mujoco.mjtState.mjSTATE_FULLPHYSICS.value
  )
  ncontrol = mujoco.mj_stateSize(model[0], control_spec)
  nv = model[0].nv
  nsensordata = model[0].nsensordata
  for m in model[1:]:
    if (
        nstate
        != mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
        or ncontrol != mujoco.mj_stateSize(m, control_spec)
        or nv != m.nv
        or nsensordata != m.nsensordata
    ):
      raise ValueError('models are not compatible')

  # check trailing dimensions
  _check_trailing_dimension(nstate, initial_state=initial_state, state=state)
  _check_trailing_dimension(ncontrol, control=control)
  _check_trailing_dimension(nv, initial_warmstart=initial_warmstart)
  _check_trailing_dimension(nsensordata, sensordata=sensordata)

  # tile input arrays/lists if required (singleton expansion)
  model = model * nroll if len(model) == 1 else model
  initial_state = _tile_if_required(initial_state, nroll)
  initial_warmstart = _tile_if_required(initial_warmstart, nroll)
  control = _tile_if_required(control, nroll, nstep)

  # allocate output if not provided
  if state is None:
    state = np.empty((nroll, nstep, nstate))
  if sensordata is None:
    sensordata = np.empty((nroll, nstep, nsensordata))

  # call rollout
  _rollout.rollout(
      model,
      data,
      nstep,
      control_spec,
      initial_state,
      initial_warmstart,
      control,
      state,
      sensordata,
      chunk_size,
  )

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
    **kwargs: List of arrays which should all have the same size (or 1) along
      dimension dim.

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
