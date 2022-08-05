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

#ifndef MUJOCO_PYTHON_CALLBACKS_H_
#define MUJOCO_PYTHON_CALLBACKS_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <type_traits>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "structs.h"
#include "raw.h"
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace mujoco::python {
namespace {
namespace py = ::pybind11;

[[noreturn]] void EscapeWithPythonException() {
  mju_error("Python exception raised");
  std::terminate();  // not actually reachable, mju_error doesn't return
}

template <typename T, typename U>
using enable_if_not_const_t =
    std::enable_if_t<std::is_same_v<std::remove_const_t<T>, T>, U>;

bool IsCallable(py::handle h) {
  static PyObject* const is_callable = []() -> PyObject* {
    try{
      PyObject* o = py::eval("callable").ptr();
      Py_XINCREF(o);
      return o;
    } catch (const py::error_already_set&) {
      return nullptr;
    }
  }();

  if (!is_callable) {
    throw UnexpectedError("cannot find `callable`");
  }

  return py::handle(is_callable)(h).cast<bool>();
}
// MuJoCo passes raw mjModel* and mjData* as arguments to callbacks, but Python
// callables expect the corresponding MjWrapper objects. To avoid creating new
// wrappers each time we enter callbacks, we instead maintain a global lookup
// table that associates raw MuJoCo struct pointers back to the pointers to
// their corresponding wrappers.
template <typename Raw>
MjWrapper<Raw>* MjWrapperLookupRaw(Raw* ptr) {
  using LookupFnType = MjWrapper<Raw>* (Raw*);
  static LookupFnType* const lookup = []() -> LookupFnType* {
    py::gil_scoped_acquire gil;
    auto m = py::module_::import("mujoco._structs");
    pybind11::handle builtins(PyEval_GetBuiltins());
    if (!builtins.contains(MjWrapper<Raw>::kFromRawPointer)) {
      return nullptr;
    } else {
      try {
        return reinterpret_cast<LookupFnType*>(
            builtins[MjWrapper<Raw>::kFromRawPointer]
                .template cast<std::uintptr_t>());
      } catch (const py::cast_error&) {
        return nullptr;
      }
    }
  }();

  MjWrapper<Raw>* wrapper = nullptr;
  if (lookup) {
    wrapper = lookup(ptr);
  } else {
    {
      py::gil_scoped_acquire gil;
      PyErr_SetString(
          UnexpectedError::GetPyExc(),
          "_structs module did not register its raw pointer lookup functions");
    }
  }

  if (!wrapper) {
    {
      py::gil_scoped_acquire gil;
      PyErr_SetString(
          UnexpectedError::GetPyExc(),
          "cannot find the corresponding wrapper for the raw mjStruct");
    }
  } else {
    return wrapper;
  }

  EscapeWithPythonException();
}

template <typename Raw>
enable_if_not_const_t<Raw, py::handle> MjWrapperLookup(Raw* ptr) {
  MjWrapper<Raw>* wrapper = MjWrapperLookupRaw(ptr);

  // Now we find the existing Python instance of our wrapper.
  // TODO(stunya): Figure out a way to do this without invoking py::detail.
  {
    py::gil_scoped_acquire gil;
    const auto [src, type] =
        py::detail::type_caster_base<MjWrapper<Raw>>::src_and_type(wrapper);
    if (type) {
      py::handle instance =
          py::detail::find_registered_python_instance(wrapper, type);
      if (!instance) {
        if (!PyErr_Occurred()) {
          PyErr_SetString(
              UnexpectedError::GetPyExc(),
              "cannot find the Python instance of the MjWrapper");
        }
      } else {
        return instance;
      }
    } else {
      if (!PyErr_Occurred()) {
        PyErr_SetString(
            UnexpectedError::GetPyExc(),
            "MjWrapper type isn't registered with pybind11");
      }
    }
  }

  EscapeWithPythonException();
}

template <typename Raw>
const py::handle MjWrapperLookup(const Raw* ptr) {
  return MjWrapperLookup(const_cast<Raw*>(ptr));
}

template <typename Return, typename... Args>
Return
CallPyCallback(const char* name, PyObject* py_callback, Args... args) {
  {
    py::gil_scoped_acquire gil;
    if (!py_callback) {
      std::ostringstream msg;
      PyErr_SetString(UnexpectedError::GetPyExc(), msg.str().c_str());
    } else {
      py::handle callback(py_callback);
      try {
        if constexpr (std::is_void_v<Return>) {
          callback(args...);
          return;
        } else {
          return callback(args...).template cast<Return>();
        }
      } catch (py::error_already_set& e) {
        e.restore();
      } catch (const py::cast_error&) {
        std::ostringstream msg;
        msg << name << " callback did not return ";
        if constexpr (std::is_integral_v<Return>) {
          msg << "an integer";
        } else if constexpr (std::is_floating_point_v<Return>) {
          msg << "a number";
        } else {
          msg << "the correct type";
        }
        PyErr_SetString(PyExc_TypeError, msg.str().c_str());
      }
    }
  }
  EscapeWithPythonException();
}

// If the Python object is a ctypes function pointer, returns the corresponding
// C function pointer. Otherwise, returns a null pointer.
template <typename FuncPtr>
FuncPtr GetCFuncPtr(py::handle h) {
  struct CTypes { PyObject* cfuncptr; PyObject* cast; PyObject* c_void_p; };
  static const CTypes ctypes = []() -> CTypes {
    try {
      auto m = py::module_::import("ctypes");
      PyObject* cfuncptr = m.attr("_CFuncPtr").ptr();
      PyObject* cast = m.attr("cast").ptr();
      PyObject* c_void_p = m.attr("c_void_p").ptr();
      Py_XINCREF(cfuncptr);
      Py_XINCREF(cast);
      Py_XINCREF(c_void_p);
      return {cfuncptr, cast, c_void_p};
    } catch (const py::error_already_set&) {
      return {nullptr, nullptr, nullptr};
    }
  }();

  if (!ctypes.cfuncptr) {
    throw UnexpectedError("cannot find `ctypes._CFuncPtr`");
  }
  const int is_cfuncptr = PyObject_IsInstance(h.ptr(), ctypes.cfuncptr);
  if (is_cfuncptr == -1) {
    throw py::error_already_set();
  } else if (is_cfuncptr) {
    if (!ctypes.cast) {
      throw UnexpectedError("cannot find `ctypes.cast`");
    }
    if (!ctypes.c_void_p) {
      throw UnexpectedError("cannot find `ctypes.c_void_p`");
    }
    const uintptr_t func_address =
        py::handle(ctypes.cast)(h, py::handle(ctypes.c_void_p))
            .attr("value")
            .template cast<std::uintptr_t>();
    return reinterpret_cast<FuncPtr>(func_address);
  } else {
    return nullptr;
  }
}

template <typename CFuncPtr>
void SetCallback(py::handle h, CFuncPtr py_trampoline,
                 PyObject** py_callback, CFuncPtr* mj_callback) {
  CFuncPtr cfuncptr = GetCFuncPtr<CFuncPtr>(h);
  if (h.is_none()) {
    Py_XDECREF(*py_callback);
    *py_callback = nullptr;
    *mj_callback = nullptr;
  } else if (cfuncptr) {
    Py_XDECREF(*py_callback);
    Py_INCREF(h.ptr());
    *py_callback = h.ptr();
    *mj_callback = cfuncptr;
  } else if (IsCallable(h)) {
    Py_XDECREF(*py_callback);
    Py_INCREF(h.ptr());
    *py_callback = h.ptr();
    *mj_callback = py_trampoline;
  } else {
    throw py::type_error("callback is not an Optional[Callable]");
  }
}

}  // namespace
}  // namespace mujoco::python

#endif
