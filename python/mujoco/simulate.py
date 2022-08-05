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
"""Wrap the pybind11 bound Simulate interface to provide type conversions"""

import mujoco
from mujoco import _structs
from mujoco import _simulate
import threading

import glfw
from glfw import _glfw
_simulate.setglfwdlhandle(_glfw._handle)

class Simulate(_simulate.Simulate):
  def __init__(self):
    super().__init__()

# These functions exist because the MjModel and MjData objects
# must be registered with a global table so that the MuJoCo
# callbacks can associated raw pointers with the appropriate
# MjWrapper<MjModel> and MjWrapper<MjData> instances.
# Because the _structs shared library cannot be linked to
# pybind11 simulate shared library these objects must be
# constructed from Python
model_dict = {}
def load_xml_func(filename):
  try:
    m = mujoco.MjModel.from_xml_path(filename)
  except Exception as e:
    print(e)
    raise e

  model_dict[m._address] = m
  return m._address

def load_binary_func(filename):
  m = mujoco.MjModel.from_binary_path(filename)
  model_dict[m._address] = m
  return m._address

data_dict = {}
def model_from_data_func(m):
  d = mujoco.MjData(m)
  data_dict[d._address] = d
  return d._address

def delete_m_func(m):
  if m._address in model_dict:
    del model_dict[m._address]

def delete_d_func(d):
  if d._address in data_dict:
    del data_dict[d._address]

def run_simulate_and_physics(file=None, preload_callback=None, load_callback=None, init=True, terminate=True):
  # simulate object encapsulates the UI

  simulate = Simulate()

  # init GLFW
  if init:
    glfw.init()

  if file is None: file = ''
  physics_thread = threading.Thread(target=lambda: _simulate.physics_thread(
    simulate, file,
    preload_callback, load_callback,
    load_xml_func, load_binary_func, model_from_data_func,
    delete_m_func, delete_d_func))
  physics_thread.start()

  # start simulation thread (this creates the UI)
  simulate.renderloop()
  physics_thread.join()

  if terminate:
    glfw.terminate()
