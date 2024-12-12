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
"""benchmarks for rollout function."""

import os
import time

import mujoco
from mujoco import rollout
import numpy as np

def benchmark_rollout(model_file, nthread=os.cpu_count()):
    print('\n', model_file)
    bench_steps = int(1e4) # Run approximately bench_steps per thread

    # A grid search
    nroll = [int(1e0), int(1e1), int(1e2), int(1e3)]
    nstep = [int(1e0), int(1e1), int(1e2), int(1e3)]
    nnroll, nnstep = np.meshgrid(nroll, nstep)
    nroll_nstep_grid = np.stack((nnroll.flatten(), nnstep.flatten()), axis=1)

    # Typical nroll/nstep for sysid, rl, mpc respectively
    nroll = [50, 3000, 100]
    nstep = [1000, 1, 50]
    nroll_nstep_app = np.stack((nroll, nstep), axis=1)

    nroll_nstep = np.vstack((nroll_nstep_grid, nroll_nstep_app))

    chunk_divisors = [10, 1, 2, 4, 8, 16, 32, 64, 128] # First element is the nominal divisor

    m = mujoco.MjModel.from_xml_path(model_file)
    print('nv:', m.nv)

    m_list = [m]*np.max(nroll) # models do not need to be copied
    d_list = [mujoco.MjData(m) for i in range(nthread)]

    initial_state = np.zeros((mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS),))
    mujoco.mj_getState(m, d_list[0], initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.tile(initial_state, (np.max(nroll), 1))

    for i in range(nroll_nstep.shape[0]):
      nroll_i = int(nroll_nstep[i, 0])
      nstep_i = int(nroll_nstep[i, 1])

      nbench = max(1, int(np.round(min(nthread, nroll_i) * bench_steps / nstep_i / nroll_i)))

      chunk_divisors_stats = []
      for chunk_divisor in chunk_divisors:
        times = [time.time()]
        for i in range(nbench):
          rollout.rollout(m_list[:nroll_i], d_list, initial_state[:nroll_i], skip_checks=True, nstep=nstep_i, chunk_divisor=chunk_divisor)
          times.append(time.time())
        dt = np.diff(times)
        chunk_divisors_stats.append((np.min(dt), np.max(dt), np.mean(dt), np.std(dt)))
      chunk_divisors_stats = np.array(chunk_divisors_stats)

      slowest_chunk_divisor_i = np.argmax(chunk_divisors_stats[:, 2])
      slowest_chunk_divisor = chunk_divisors[slowest_chunk_divisor_i]

      fastest_chunk_divisor_i = np.argmin(chunk_divisors_stats[:, 2])
      fastest_chunk_divisor = chunk_divisors[fastest_chunk_divisor_i]

      print('nbench: {:06d} nroll: {:04d} nstep: {:04d} '
            'mean_nom {:0.4f} mean_slow: {:0.4f} mean_fast: {:0.4f} chunk_div_slow {:03d} chunk_div_fast {:03d} fast/slow {:0.3f} fast/nom {:0.3f}'.format(
            nbench, nroll_i, nstep_i,
            chunk_divisors_stats[0, 2], # nominal chunk divisor
            chunk_divisors_stats[slowest_chunk_divisor_i, 2],
            chunk_divisors_stats[fastest_chunk_divisor_i, 2],
            slowest_chunk_divisor, fastest_chunk_divisor,
            chunk_divisors_stats[fastest_chunk_divisor_i, 2] / chunk_divisors_stats[slowest_chunk_divisor_i, 2],
            chunk_divisors_stats[fastest_chunk_divisor_i, 2] / chunk_divisors_stats[0, 2]))

if __name__ == '__main__':
  print('============================================================')
  print('small to medium models')
  print('============================================================')

  benchmark_rollout(model_file='../../../dm_control/dm_control/suite/hopper.xml')
  benchmark_rollout(model_file='../../../mujoco_menagerie/unitree_go2/scene.xml')
  benchmark_rollout(model_file='../../model/humanoid/humanoid.xml')

  print()
  print('============================================================')
  print('very large models')
  print('============================================================')
  benchmark_rollout(model_file='../../model/cards/cards.xml')
  benchmark_rollout(model_file='../../model/humanoid/humanoid100.xml')
  benchmark_rollout(model_file='../../test/benchmark/testdata/humanoid200.xml')
  # benchmark_rollout(model_file='../../model/humanoid/100_humanoids.xml')
