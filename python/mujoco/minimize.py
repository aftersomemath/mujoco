# Copyright 2024 DeepMind Technologies Limited
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
"""Nonlinear Least Squares minimization with box bounds."""

import dataclasses
import enum
import time
from typing import Callable, List, Optional, TextIO, Tuple, Union

import mujoco
import numpy as np


class Verbosity(enum.Enum):
  SILENT = 0
  FINAL = 1
  ITER = 2
  FULLITER = 3


class Status(enum.Enum):
  FACTORIZATION_FAILED = enum.auto()
  NO_IMPROVEMENT = enum.auto()
  MAX_ITER = enum.auto()
  DX_TOL = enum.auto()


_STATUS_MESSAGE = {
    Status.FACTORIZATION_FAILED: 'factorization failed.',
    Status.NO_IMPROVEMENT: 'insufficient reduction.',
    Status.MAX_ITER: 'maximum iterations reached.',
    Status.DX_TOL: 'norm(dx) < tol.',
}


@dataclasses.dataclass(frozen=True)
class IterLog:
  """Log of a single iteration of the non-linear least-squares solver.

  Attributes:
    candidate: Value of the decision variable at the beginning this iteration.
    objective: Value of the objective at the candidate.
    reduction: Reduction of the objective during this iteration.
    regularizer: Value of the regularizer used for this iteration.
    residual: Optional value of the residual at the candidate.
    jacobian: Optional value of the Jacobian at the candidate.
    step: Optional change in decision variable during this iteration.
  """

  candidate: np.ndarray
  objective: np.float64
  reduction: np.float64
  regularizer: np.float64
  residual: Optional[np.ndarray] = None
  jacobian: Optional[np.ndarray] = None
  step: Optional[np.ndarray] = None


def jacobian_fd(
    residual: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    r: np.ndarray,
    eps: np.float64,
    central: bool,
    jn_res: int,
    bounds: Optional[List[np.ndarray]] = None,
    process_pool = None,
):
  """Finite-difference Jacobian of a residual function.

  Args:
    residual: function that returns the residual for a given point.
    x: point at which to evaluate the Jacobian.
    r: residual at x.
    eps: finite-difference step size.
    central: whether to use central differences.
    jn_res: number of jacobian residual evaluations so far.
    bounds: optional pair of lower and upper bounds.

  Returns:
    jac: Jacobian of the residual at x.
    jn_res: updated number of jacobian residual evaluations.

  """

  assert bounds is not None
  assert central == False

  residual_args = []


  nx = x.size
  nr = r.size
  jac = np.zeros((nr, nx))
  xh = x.copy()
  if bounds is None:
    # No bounds, simple forward or central differencing.
    for i in range(nx):
      xh[i] = x[i] + eps
      rp = residual(xh)
      if central:
        xh[i] = x[i] - eps
        rm = residual(xh)
        jac[:, i] = (rp - rm) / (2*eps)
      else:
        jac[:, i] = (rp - r) / eps
      xh[i] = x[i]
    jn_res += 2*nx if central else nx
  else:
    lower, upper = bounds
    midpoint = 0.5 * (upper - lower)
    for i in range(nx):
      # Scale eps, don't cross bounds.
      eps_i = eps * (upper[i] - lower[i])
      if central:
        # Use central differencing if away from bounds.
        if x[i] - eps_i < lower[i]:
          # Near lower bound, use forward.
          xh[i] = x[i] + eps_i
          rp = residual(xh)
          jac[:, i] = (rp - r) / eps_i
          jn_res += 1
        elif x[i] + eps_i > upper[i]:
          # Near upper bound, use backward.
          xh[i] = x[i] - eps_i
          rm = residual(xh)
          jac[:, i] = (r - rm) / eps_i
          jn_res += 1
        else:
          # Use central.
          xh[i] = x[i] + eps_i
          rp = residual(xh)
          xh[i] = x[i] - eps_i
          rm = residual(xh)
          jac[:, i] = (rp - rm) / (2*eps_i)
          jn_res += 2
      else: # This is the only case that should evaluate because of above
        # Below midpoint use forward differencing, otherwise backward.
        if x[i] < midpoint[i]:
          xh[i] = x[i] + eps_i

          residual_args.append((i, True, eps_i, np.copy(xh)))
          # rp = residual(xh)
          # jac[:, i] = (rp - r) / eps_i
        else:
          xh[i] = x[i] - eps_i

          residual_args.append((i, False, eps_i, np.copy(xh)))
          # rm = residual(xh)
          # jac[:, i] = (r - rm) / eps_i
        jn_res += 1
      # Reset.
      xh[i] = x[i]

  if process_pool is None:
    for i, eps_sign, eps_i, xh in residual_args:
      if eps_sign:
        rp = residual(xh)
        jac[:, i] = (rp - r) / eps_i
      else:
        rm = residual(xh)
        jac[:, i] = (r - rm) / eps_i
  else:
    rpm_list = process_pool.map(residual, [arg[3] for arg in residual_args])

    assert len(rpm_list) == len(residual_args)
    for (i, eps_sign, eps_i, _), rpm in zip(residual_args, rpm_list):
      if eps_sign:
        rp = rpm
        jac[:, i] = (rp - r) / eps_i
      else:
        rm = rpm
        jac[:, i] = (r - rm) / eps_i

  return jac, jn_res


def least_squares(
    x0: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    bounds: Optional[List[np.ndarray]] = None,
    jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    eps: float = 1e-6,
    central: bool = False,
    mu_min: float = 1e-6,
    mu_max: float = 1e8,
    mu_factor: float = 10.0**0.1,
    tol: float = 1e-7,
    max_iter: int = 100,
    verbose: Union[Verbosity, int] = Verbosity.ITER,
    output: Optional[TextIO] = None,
    process_pool = None,
) -> Tuple[np.ndarray, List[IterLog]]:
  """Nonlinear Least Squares minimization with box bounds.

  Args:
    x0: initial guess
    residual: function that returns the residual for a given point x.
    bounds: optional pair of lower and upper bounds on the solution.
    jacobian: optional function that returns Jacobian of the residual at a given
      point and residual. If not given, `residual` will be finite-differenced.
    eps: perurbation used for automatic finite-differencing.
    central: whether to use central differences.
    mu_min: minimum value of the regularizer.
    mu_max: maximum value of the regularizer.
    mu_factor: factor increasing or decreasing the regularizer.
    tol: termination tolerance on the step size.
    max_iter: maximum number of iterations.
    verbose: verbosity level.
    output: optional file or StringIO to which to print messages.

  Returns:
    x: best solution found
    trace: sequence of solution iterates.
  """
  t_start_total = time.time()

  # Convert verbosity to int.
  verbose = Verbosity(verbose).value

  # Constant for Armijo's sufficient-reduction rule
  armijo_c1 = 1e-2

  # Initialize locals.
  status = Status.MAX_ITER
  i = 0
  x = x0.astype(np.float64)
  n = x.size
  xnew = np.zeros((n,))
  dx = np.zeros((n,))
  scratch = np.zeros((n, n + 7))
  eps = np.float64(eps)
  mu = np.float64(0.0)  # Optimistically start with no regularization.
  n_reduc = 0  # Number of sequential mu reductions.

  # Initialize logging.
  trace = []
  last_n_res = 0
  last_n_jres = 0
  last_n_jac = 0
  n_res = 0
  n_jres = 0
  n_jac = 0
  t_res = 0.0
  t_jac = 0.0

  if mu_factor <= 1:
    raise ValueError('mu_factor must be > 1.')

  # Decrease mu agressively: sequential decreases grow exponentially.
  def decrease_mu(mu, n_reduc):
    dmu = (1/mu_factor) ** (2**n_reduc)
    mu = 0.0 if mu * dmu < mu_min else mu * dmu
    n_reduc += 1
    return mu, n_reduc

  # Increase mu carefully: always increase by mu_factor.
  def increase_mu(mu):
    mu = max(mu_min, mu_factor * mu)
    n_reduc = 0  # Reset n_reduc.
    return mu, n_reduc

  if bounds is not None:
    # Checks bounds.
    if len(bounds) != 2:
      raise ValueError('bounds must have 2 elements.')
    if bounds[0].size != n or bounds[1].size != n:
      raise ValueError('bounds must have the same size as x0.')
    if not np.all(np.isfinite(bounds[0])) or not np.all(np.isfinite(bounds[1])):
      raise ValueError('bounds must be finite.')
    if not np.all(bounds[0] < bounds[1]):
      raise ValueError('bounds[0] must be smaller than bounds[1].')
    # Clip.
    np.clip(x, bounds[0], bounds[1], out=x)

  # Check for NaNs.
  if not np.all(np.isfinite(x)):
    raise ValueError('x0 must be finite.')

  # Get initial residual.
  t_start = time.time()
  r = residual(x)
  rnew = r
  t_res += time.time() - t_start
  n_res += 1

  if r.dtype != np.float64:
    raise ValueError('residual function must return float64 arrays.')

  # Minimize.
  for i in range(max_iter):
    if status != Status.MAX_ITER:
      break

    # Get objective y.
    y = 0.5 * r.dot(r)

    # Get Jacobian jac.
    t_start = time.time()
    if jacobian is None:
      jac, n_jres = jacobian_fd(residual, x, r, eps, central, n_jres, bounds, process_pool)
      t_jac += time.time() - t_start
      n_jac += 1
    else:
      jac = jacobian(x, r)
      t_jac += time.time() - t_start
      n_jac += 1

    # Get gradient, Gauss-Newton Hessian.
    grad = jac.T @ r
    hess = jac.T @ jac

    # Bounds relative to x
    dlower = None if bounds is None else bounds[0] - x
    dupper = None if bounds is None else bounds[1] - x

    # Find reduction satisfying Armijo's rule.
    armijo = -1
    reduction = 0.0
    while armijo < 0:
      # Increase mu until factorizable.
      factorizable = False
      while not factorizable:
        n_free = mujoco.mju_boxQP(
            dx, scratch, None, hess + mu * np.eye(n), grad, dlower, dupper
        )
        if n_free >= 0:
          factorizable = True
        elif mu >= mu_max:
          status = Status.FACTORIZATION_FAILED
          break
        else:
          mu, n_reduc = increase_mu(mu)

      if status != Status.MAX_ITER:
        break

      # New candidate, residual.
      xnew = x + dx
      t_start = time.time()
      rnew = residual(xnew)
      t_res += time.time() - t_start
      n_res += 1

      # New objective, evaluate reduction.
      ynew = 0.5 * rnew.dot(rnew)
      reduction = y - ynew
      armijo = reduction + armijo_c1*grad.dot(dx)

      if armijo < 0:
        if mu >= mu_max:
          status = Status.NO_IMPROVEMENT
          break
        mu, n_reduc = increase_mu(mu)

    if status != Status.MAX_ITER:
      break

    # Compute reduction ratio.
    expected_reduction = -(grad.dot(dx) + 0.5 * dx.T @ hess @ dx)
    reduction_ratio = 0.0
    if expected_reduction <= 0:
      if verbose > Verbosity.SILENT.value:
        if expected_reduction == 0:
          print('Zero expected reduction: exact minimum found?', file=output)
        elif expected_reduction < 0:
          print('Negative expected reduction: should not occur.', file=output)
    else:
      reduction_ratio = reduction / expected_reduction

    # Iteration message.
    dx_norm = np.linalg.norm(dx)
    if verbose >= Verbosity.ITER.value:
      logmu = np.log10(mu) if mu > 0 else -np.inf
      message = (
          f'iter: {i:<3d}  y: {y:<9.4g}  log10mu: {logmu:>4.1f}  '
          f'ratio: {reduction_ratio:<7.2g}  '
          f'dx: {dx_norm:<7.2g}  reduction: {reduction:<7.2g}  '
          f'Res evals: {n_res-last_n_res:d}  Jac res evals {n_jres-last_n_jres:d}  Jac evals {n_jac-last_n_jac:d}'
      )
      print(message, file=output)
      last_n_res = n_res
      last_n_jres = n_jres
      last_n_jac = n_jac

    # Append log to trace.
    log = IterLog(candidate=x, objective=y, reduction=reduction, regularizer=mu)
    if verbose >= Verbosity.FULLITER.value:
      log = dataclasses.replace(log, residual=r, jacobian=jac, step=dx)
    trace.append(log)

    # Check for success.
    if dx_norm < tol:
      status = Status.DX_TOL
      break

    # Modify regularizer like in (Bazaraa, Sherali, and Shetty)
    if reduction_ratio > 0.75:
      mu, n_reduc = decrease_mu(mu, n_reduc)
    elif reduction_ratio < 0.25:
      mu, n_reduc = increase_mu(mu)

    # Accept proposal.
    x = xnew
    r = rnew

  # Final Iteration message.
  if status != Status.DX_TOL and verbose >= Verbosity.ITER.value:
    dx_norm = np.linalg.norm(dx)
    logmu = np.log10(mu) if mu > 0 else -np.inf
    message = (
        f'iter: {i:<3d}  y: {y:<9.4g}  log10mu: {logmu:>4.1f}  '
        f'ratio: {reduction_ratio:<7.2g}  '
        f'dx: {dx_norm:<7.2g}  reduction: {reduction:<7.2g}  '
        f'Res evals: {n_res-last_n_res:d}  Jac res evals {n_jres-last_n_jres:d}  Jac evals {n_jac-last_n_jac:d}'
    )
    print(message, file=output)
    last_n_res = n_res
    last_n_jres = n_jres
    last_n_jac = n_jac

  # Append final log to trace.
  # Note: unlike other iter logs, this is at the end point.
  yfinal = 0.5 * r.dot(r)
  red = np.float64(0.0)
  log = IterLog(candidate=x, objective=yfinal, reduction=red, regularizer=mu)
  trace.append(log)

  # Print final diagnostics.
  if verbose > Verbosity.SILENT.value:
    message = f'Terminated after {i} iterations: '
    message += _STATUS_MESSAGE[status]
    message += f' y: {yfinal:<.4g}, Residual evals: {n_res:d}'
    if n_jres > 0:
      message += f', Jacobian residual evals: {n_jres:d}'
    if n_jac > 0:
      message += f', Jacobian evals: {n_jac:d}'
    print(message, file=output)

    time_total = time.time() - t_start_total
    if time_total > 0:
      r_percent = 100 * t_res / time_total
      time_scale = 1 if time_total > 1 else 1000
      time_units = 's' if time_total > 1 else 'ms'
      message = f'total time {time_scale * time_total:<.1f}{time_units}'
      message += f' of which residual {r_percent:<.1f}%'
      if t_jac > 0:
        jac_percent = 100 * t_jac / time_total
        message += f' Jacobian {jac_percent:<.1f}%'
      print(message, file=output)

  return x, trace
