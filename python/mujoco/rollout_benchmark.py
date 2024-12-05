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
import threading
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

def benchmark_rollout(model_file='../../test/benchmark/testdata/humanoid200.xml'):
    nthread = 24
    nroll = [int(1e0), int(1e1), int(1e2)]
    nstep = [int(1e0), int(1e1), int(1e2), int(2e2)]

    print('making structures')
    m = mujoco.MjModel.from_xml_path(model_file)
    m_list = [m]*nroll[-1] # models do not need to be copied
    d_list = [mujoco.MjData(m) for i in range(nthread)]

    initial_state = np.zeros((mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS),))
    mujoco.mj_getState(m, d_list[0], initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_state = np.tile(initial_state, (nroll[-1], 1))

    print('initializing thread pools')
    pt = PythonThreading(m_list[0], len(d_list))

    print('running benchmark')
    for nroll_i in nroll:
        for nstep_i in nstep:
            nt_res = timeit.timeit(lambda: rollout.rollout(m_list[:nroll_i], d_list, initial_state[:nroll_i], nstep=nstep_i), number=10)
            pt_res = timeit.timeit(lambda: pt.run(m_list[:nroll_i], initial_state[:nroll_i], nstep_i), number=10)
            print('{:03d} {:03d} {:0.3f} {:0.3f} {:0.3f}'.format(nroll_i, nstep_i, nt_res, pt_res, nt_res / pt_res))

    # Generate plots

if __name__ == '__main__':
  benchmark_rollout()
