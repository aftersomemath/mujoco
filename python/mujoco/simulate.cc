// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstdio>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <mujoco/mujoco.h>
#include <mujoco/simulate.h>
#include "callbacks.h"
#include "raw.h"
#include "structs.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;

PyObject* py_preload_callback = nullptr;
void PyPreloadCallback(const mujoco::raw::MjModel* m, mujoco::raw::MjData* d) {
  if (m && d) {
    CallPyCallback<void>("preload_callback", py_preload_callback,
                         MjWrapperLookup(m), MjWrapperLookup(d));
  }
}

PyObject* py_load_callback = nullptr;
void PyLoadCallback(const raw::MjModel* m, raw::MjData* d) {
  CallPyCallback<void>("load_callback", py_load_callback,
                       MjWrapperLookup(m), MjWrapperLookup(d));
}

PyObject* py_load_xml_callback = nullptr;
raw::MjModel* PyLoadXMLCallback(const char* filename) {
  std::uintptr_t m_uintptr = CallPyCallback<std::uintptr_t>(
    "load_xml_callback",py_load_xml_callback, std::string(filename));
  return reinterpret_cast<raw::MjModel*>(m_uintptr);
}

PyObject* py_load_binary_callback = nullptr;
raw::MjModel* PyLoadBinaryCallback(const char* filename) {
  std::uintptr_t m_uintptr = CallPyCallback<std::uintptr_t>(
    "load_binary_callback", py_load_binary_callback, std::string(filename));
  return reinterpret_cast<raw::MjModel*>(m_uintptr);
}

PyObject* py_data_from_model_callback = nullptr;
raw::MjData* PyDataFromModelCallback(const raw::MjModel* m) {
  std::uintptr_t d_uintptr = CallPyCallback<std::uintptr_t>(
    "data_from_model_callback", py_data_from_model_callback, MjWrapperLookup(m));
  return reinterpret_cast<raw::MjData*>(d_uintptr);
}

PyObject* py_delete_m_callback = nullptr;
void PyDeleteMCallback(const raw::MjModel* m) {
    CallPyCallback<void>("delete_m", py_delete_m_callback, MjWrapperLookup(m));
}

PyObject* py_delete_d_callback = nullptr;
void PyDeleteDCallback(const raw::MjData* d) {
    CallPyCallback<void>("delete_d", py_delete_d_callback, MjWrapperLookup(d));
}

PYBIND11_MODULE(_simulate, pymodule) {
  namespace py = ::pybind11;

  py::class_<mujoco::Simulate>(pymodule, "Simulate")
    .def(py::init<>())
    .def("renderloop",
      [](mujoco::Simulate& simulate) {
        simulate.renderloop();
      },
      py::call_guard<py::gil_scoped_release>()
    );

  pymodule.def("setglfwdlhandle", [](std::uintptr_t dlhandle) { mujoco::setglfwdlhandle(reinterpret_cast<void*>(dlhandle)); });

  pymodule.def(
    "physics_thread",
    [](mujoco::Simulate& simulate, std::string& filename,
       py::handle preload_callback, py::handle load_callback,
       py::handle load_xml_callback, py::handle load_binary_callback,
       py::handle data_from_model_callback, py::handle delete_m_callback,
       py::handle delete_d_callback) {
      mjfGeneric preload_c_func;
      SetCallback(preload_callback, PyPreloadCallback, &py_preload_callback, &preload_c_func);

      mjfGeneric load_c_func;
      SetCallback(load_callback, PyLoadCallback, &py_load_callback, &load_c_func);

      simulate_model_load_func load_xml_c_func;
      SetCallback(load_xml_callback, PyLoadXMLCallback, &py_load_xml_callback, &load_xml_c_func);

      simulate_model_load_func load_binary_c_func;
      SetCallback(load_binary_callback, PyLoadBinaryCallback, &py_load_binary_callback, &load_binary_c_func);

      simulate_data_from_model_func data_from_model_c_func;
      SetCallback(data_from_model_callback, PyDataFromModelCallback, &py_data_from_model_callback, &data_from_model_c_func);

      simulate_delete_model_func delete_m_c_func;
      SetCallback(delete_m_callback, PyDeleteMCallback, &py_delete_m_callback, &delete_m_c_func);

      simulate_delete_data_func delete_d_c_func;
      SetCallback(delete_d_callback, PyDeleteDCallback, &py_delete_d_callback, &delete_d_c_func);

      // Do not release the GIL until the callbacks are registered
      {
        py::gil_scoped_release release;

        const char* filename_c_str = filename.empty() ? nullptr: filename.c_str();

        mujoco::PhysicsThread(simulate, filename_c_str,
                              preload_c_func, load_c_func,
                              load_xml_c_func, load_binary_c_func,
                              data_from_model_c_func,
                              delete_m_c_func, delete_d_c_func);
      }
    }
  );
}

}  // namespace

}
