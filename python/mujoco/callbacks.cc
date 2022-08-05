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

#include "callbacks.h"

namespace mujoco::python {
namespace {
namespace py = ::pybind11;

static PyObject* py_mju_user_warning = nullptr;
static void PyMjuUserWarning(const char* msg) {
  CallPyCallback<void>("mju_user_warning", py_mju_user_warning, msg);
}

// We only support ctypes function pointers for these.
// The PyObject* are only here so that we can return the ctypes pointers back
// through the getters.
static PyObject* py_mju_user_malloc = nullptr;
static PyObject* py_mju_user_free = nullptr;

static PyObject* py_mjcb_passive = nullptr;
static void PyMjcbPassive(const raw::MjModel* m, raw::MjData* d) {
  CallPyCallback<void>("mjcb_passive", py_mjcb_passive,
                        MjWrapperLookup(m), MjWrapperLookup(d));
}

static PyObject* py_mjcb_control = nullptr;
static void PyMjcbControl(const raw::MjModel* m, raw::MjData* d) {
  CallPyCallback<void>("mjcb_control", py_mjcb_control,
                         MjWrapperLookup(m), MjWrapperLookup(d));
}

static PyObject* py_mjcb_contactfilter = nullptr;
static int PyMjcbContactfilter(
    const raw::MjModel* m, raw::MjData* d, int geom1, int geom2) {
  return CallPyCallback<int>("mjcb_contactfilter", py_mjcb_contactfilter,
                               MjWrapperLookup(m), MjWrapperLookup(d),
                               geom1, geom2);
}

static PyObject* py_mjcb_sensor = nullptr;
static void
PyMjcbSensor(const raw::MjModel* m, raw::MjData* d, int stage) {
  CallPyCallback<void>("mjcb_sensor", py_mjcb_sensor,
                         MjWrapperLookup(m), MjWrapperLookup(d), stage);
}

static PyObject* py_mjcb_time = nullptr;
static mjtNum PyMjcbTime() {
  return CallPyCallback<mjtNum>("mjcb_time", py_mjcb_time);
}

static PyObject* py_mjcb_act_dyn = nullptr;
static mjtNum
PyMjcbActDyn(const raw::MjModel* m, const raw::MjData* d, int id) {
  return CallPyCallback<mjtNum>("mjcb_act_dyn", py_mjcb_act_dyn,
                                MjWrapperLookup(m), MjWrapperLookup(d), id);
}

static PyObject* py_mjcb_act_gain = nullptr;
static mjtNum
PyMjcbActGain(const raw::MjModel* m, const raw::MjData* d, int id) {
  return CallPyCallback<mjtNum>("mjcb_act_gain", py_mjcb_act_gain,
                                MjWrapperLookup(m), MjWrapperLookup(d), id);
}

static PyObject* py_mjcb_act_bias = nullptr;
static mjtNum
PyMjcbActBias(const raw::MjModel* m, const raw::MjData* d, int id) {
  return CallPyCallback<mjtNum>("mjcb_act_bias", py_mjcb_act_bias,
                                MjWrapperLookup(m), MjWrapperLookup(d), id);
}

py::object GetCallback(PyObject* py_callback) {
  if (!py_callback) {
    return py::none();
  }
  return py::reinterpret_borrow<py::object>(py_callback);
}

PYBIND11_MODULE(_callbacks, pymodule) {
  // Setters
  pymodule.def("set_mju_user_warning", [](py::handle h) {
    SetCallback(h, PyMjuUserWarning, &py_mju_user_warning, &::mju_user_warning);
  });
  pymodule.def("set_mju_user_malloc", [](py::handle h) {
    if (h.is_none()) {
      Py_XDECREF(py_mju_user_malloc);
      py_mju_user_malloc = nullptr;
    } else {
      auto* cfuncptr = GetCFuncPtr<decltype(::mju_user_malloc)>(h);
      if (!cfuncptr) {
        throw py::type_error("mju_user_malloc must be a C function pointer");
      }
      Py_XDECREF(py_mju_user_malloc);
      Py_XINCREF(h.ptr());
      py_mju_user_malloc = h.ptr();
      ::mju_user_malloc = cfuncptr;
    }
  });
  pymodule.def("set_mju_user_free", [](py::handle h) {
    if (h.is_none()) {
      Py_XDECREF(py_mju_user_free);
      py_mju_user_free = nullptr;
    } else {
      auto* cfuncptr = GetCFuncPtr<decltype(::mju_user_free)>(h);
      if (!cfuncptr) {
        throw py::type_error("mju_user_free must be a C function pointer");
      }
      Py_XDECREF(py_mju_user_free);
      Py_XINCREF(h.ptr());
      py_mju_user_free = h.ptr();
      ::mju_user_free = cfuncptr;
    }
  });
  pymodule.def("set_mjcb_passive", [](py::handle h) {
    SetCallback(h, PyMjcbPassive, &py_mjcb_passive, &::mjcb_passive);
  });
  pymodule.def("set_mjcb_control", [](py::handle h) {
    SetCallback(h, PyMjcbControl, &py_mjcb_control, &::mjcb_control);
  });
  pymodule.def("set_mjcb_contactfilter", [](py::handle h) {
    SetCallback(h, PyMjcbContactfilter,
                &py_mjcb_contactfilter, &::mjcb_contactfilter);
  });
  pymodule.def("set_mjcb_sensor", [](py::handle h) {
    SetCallback(h, PyMjcbSensor, &py_mjcb_sensor, &::mjcb_sensor);
  });
  pymodule.def("set_mjcb_time", [](py::handle h) {
    SetCallback(h, PyMjcbTime, &py_mjcb_time, &::mjcb_time);
  });
  pymodule.def("set_mjcb_act_dyn", [](py::handle h) {
    SetCallback(h, PyMjcbActDyn, &py_mjcb_act_dyn, &::mjcb_act_dyn);
  });
  pymodule.def("set_mjcb_act_gain", [](py::handle h) {
    SetCallback(h, PyMjcbActGain, &py_mjcb_act_gain, &::mjcb_act_gain);
  });
  pymodule.def("set_mjcb_act_bias", [](py::handle h) {
    SetCallback(h, PyMjcbActBias, &py_mjcb_act_bias, &::mjcb_act_bias);
  });

  // Getters
  pymodule.def("get_mju_user_warning", []() {
    return GetCallback(py_mju_user_warning);
  });
  pymodule.def("get_mju_user_malloc", []() {
    return GetCallback(py_mju_user_malloc);
  });
  pymodule.def("get_mju_user_free", []() {
    return GetCallback(py_mju_user_free);
  });
  pymodule.def("get_mjcb_passive", []() {
    return GetCallback(py_mjcb_passive);
  });
  pymodule.def("get_mjcb_control", []() {
    return GetCallback(py_mjcb_control);
  });
  pymodule.def("get_mjcb_contactfilter", []() {
    return GetCallback(py_mjcb_contactfilter);
  });
  pymodule.def("get_mjcb_sensor", []() {
    return GetCallback(py_mjcb_sensor);
  });
  pymodule.def("get_mjcb_time", []() {
    return GetCallback(py_mjcb_time);
  });
  pymodule.def("get_mjcb_act_dyn", []() {
    return GetCallback(py_mjcb_act_dyn);
  });
  pymodule.def("get_mjcb_act_gain", []() {
    return GetCallback(py_mjcb_act_gain);
  });
  pymodule.def("get_mjcb_act_bias", []() {
    return GetCallback(py_mjcb_act_bias);
  });
}  // PYBIND11_MODULE
}  // namespace
}  // namespace mujoco::python
