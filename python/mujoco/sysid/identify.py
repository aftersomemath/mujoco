"""Default Identify for basic identification + reporting"""

import os
import pathlib
import pickle
from typing import Any, List, Literal, Mapping, Optional

from absl import logging
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco.sysid import model_modifieruv
from mujoco.sysid import parameter
from mujoco.sysid import signal_modifier
from mujoco.sysid import sysid
from mujoco.sysid import timeseries


def run_experiment(
    experiment_results_folder: str,
    models_sequences,
    params: parameter.ParameterDict,
    cfg,
    seed: int = 1,
    n_threads: Optional[int] = os.cpu_count(),
    sample_initial_solution: bool = True,
    optimizer: Literal["scipy", "mujoco"] = "scipy",
    optimizer_kwargs: Mapping[str, Any] = {},
    validate: bool = False,
    x0_path: str = None,
    modify_residual = None,
    title: str = "MuJoCo Sysid Report",
    enabled_observations = []
) -> None:
  if seed is None:
    rng = np.random.default_rng()
  else:
    rng = np.random.default_rng(seed)

  rng = np.random.default_rng()
  x_nominal = params.as_vector()
  used_sensor_indices = signal_modifier.get_sensor_indices(
      models_sequences[0].gt_model, cfg.sensors_enabled
  )

  # Sample initial solution.
  if x0_path is not None:
    x0_params = parameter.ParameterDict.load_from_disk(
        os.path.join(x0_path, "params_x_hat.yaml")
    )
    x0 = x0_params.as_vector()
  elif sample_initial_solution:
    logging.info(f"Initial solution: random")
    x0 = params.sample(rng=rng)
  else:
    logging.info("Inital solution: nominal")
    x0 = x_nominal

  params.randomize()

  residual_fn = sysid.build_residual_fn(
    models_sequences=models_sequences,
    modify_residual=modify_residual,
    enabled_observations=enabled_observations,
  )

  if validate:
    with open(os.path.join(x0_path, "results.pkl"), "rb") as handle:
      opt_result = pickle.load(handle)
  else:
    # Optimize.
    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_fn,
        optimizer=optimizer,
        **optimizer_kwargs,
    )

  if not validate:
    sysid.save_results(
        experiment_results_folder=experiment_results_folder,
        models_sequences=models_sequences,
        initial_params=params,
        opt_params=opt_params,
        opt_result=opt_result,
        residual_fn=residual_fn,
    )
    sysid.default_report(
        models_sequences=models_sequences,
        initial_params=params,
        opt_params=opt_params,
        build_model=model_modifier.apply_param_modifiers,
        residual_fn=residual_fn,
        opt_result=opt_result,
        title=title,
        save_path = pathlib.Path(experiment_results_folder)
    )
