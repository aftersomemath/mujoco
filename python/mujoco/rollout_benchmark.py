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

import concurrent.futures
import os
import threading
import time
import timeit

import mujoco
from mujoco import rollout
import numpy as np

class PythonThreading:
  def __init__(self, m_example, num_workers):
    self.m_example = m_example
    self.num_workers = num_workers
    self.thread_local = threading.local()
    self.executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self.num_workers, initializer=self.thread_initializer)

  def thread_initializer(self):
    self.thread_local.data = mujoco.MjData(self.m_example)

  def call_rollout(self, model_list, initial_state, nstep):
    rollout.rollout(model_list, [self.thread_local.data], initial_state,
                    skip_checks=True,
                    nstep=nstep)

  def run(self, model_list, initial_state, nstep):
    nroll = len(model_list)

    # Divide jobs evenly across threads (as in test)
    # better for very wide rollouts
    # chunk_size = max(1, nroll // self.num_workers)

    # Divide jobs across threads with a chunksize 1/10th of the even amount
    # new strategy, helps with load balancing
    chunk_size = max(1, nroll // (10 * self.num_workers))

    nfulljobs = nroll // chunk_size;
    chunk_remainder = nroll % chunk_size;
    njobs = nfulljobs
    if (chunk_remainder > 0): njobs += 1

    chunks = [] # a list of tuples, one per worker
    for i in range(nfulljobs):
      chunks.append((model_list[i*chunk_size:(i+1)*chunk_size],
                     initial_state[i*chunk_size:(i+1)*chunk_size],
                     nstep))
    if chunk_remainder > 0:
      chunks.append((model_list[nfulljobs*chunk_size:],
                     initial_state[nfulljobs*chunk_size:],
                     nstep))

    futures = []
    for chunk in chunks:
      futures.append(self.executor.submit(self.call_rollout, *chunk))
    for future in concurrent.futures.as_completed(futures):
      future.result()

def benchmark_rollout(model_file, nthread=os.cpu_count()):
    print()
    print(model_file)
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

    # nroll_nstep = np.vstack((nroll_nstep_grid, nroll_nstep_app))
    nroll_nstep = nroll_nstep_app

    m = mujoco.MjModel.from_xml_path(model_file)
    print('nv:', m.nv)

    m_list = [m]*np.max(nroll) # models do not need to be copied
    d_list = [mujoco.MjData(m) for i in range(nthread)]

    initial_state = np.zeros((mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS),))
    mujoco.mj_getState(m, d_list[0], initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.tile(initial_state, (np.max(nroll), 1))

    pt = PythonThreading(m_list[0], len(d_list))

    for i in range(nroll_nstep.shape[0]):
      nroll_i = int(nroll_nstep[i, 0])
      nstep_i = int(nroll_nstep[i, 1])

      nbench = max(1, int(np.round(min(nthread, nroll_i) * bench_steps / nstep_i / nroll_i)))

      times = [time.time()]
      for i in range(nbench):
        rollout.rollout(m_list[:nroll_i], d_list, initial_state[:nroll_i], skip_checks=True, nstep=nstep_i)
        times.append(time.time())
      dt = np.diff(times)
      nt_stats = (np.min(dt), np.max(dt), np.mean(dt), np.std(dt))

      # times = [time.time()]
      # for i in range(nbench):
      #   pt.run(m_list[:nroll_i], initial_state[:nroll_i], nstep_i)
      #   times.append(time.time())
      # dt = np.diff(times)
      # pt_stats = (np.min(dt), np.max(dt), np.mean(dt), np.std(dt))

      print('nbench: {:06d} nroll: {:04d} nstep: {:04d} '
            'nt_min: {:0.4f} nt_max: {:0.4f} nt_mean: {:0.4f} nt_std: {:0.4f} '.format(
        nbench, nroll_i, nstep_i,
        *nt_stats))#, *pt_stats,
        #nt_stats[0] / pt_stats[0], nt_stats[2] / pt_stats[2]))
      time.sleep(5)

if __name__ == '__main__':
  print('============================================================')
  print('small to medium models')
  print('============================================================')

  benchmark_rollout(model_file='../../../dm_control/dm_control/suite/hopper.xml')
  benchmark_rollout(model_file='../../../mujoco_menagerie/unitree_go2/scene.xml')
  benchmark_rollout(model_file='../../model/humanoid/humanoid.xml')
  exit(0)

  print()
  print('============================================================')
  print('very large models')
  print('============================================================')
  benchmark_rollout(model_file='../../model/cards/cards.xml')
  benchmark_rollout(model_file='../../model/humanoid/humanoid100.xml')
  benchmark_rollout(model_file='../../test/benchmark/testdata/humanoid200.xml')
  # benchmark_rollout(model_file='../../model/humanoid/100_humanoids.xml')
