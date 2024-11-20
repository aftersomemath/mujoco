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

#include <iostream>
#include <optional>
#include <sstream>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "raw.h"
#include "structs.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;

// NOLINTBEGIN(whitespace/line_length)

const auto rollout_doc = R"(
Roll out open-loop trajectories from initial states, get resulting states and sensor values.

  input arguments (required):
    model              instance of MjModel
    data               associated instance of MjData
    nroll              integer, number of initial states from which to roll out trajectories
    nstep              integer, number of steps to be taken for each trajectory
    control_spec       specification of controls, ncontrol = mj_stateSize(m, control_spec)
    state0             (nroll x nstate) nroll initial state vectors,
                                        nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS)
  input arguments (optional):
    warmstart0         (nroll x nv)                   nroll qacc_warmstart vectors
    control            (nroll x nstep x ncontrol)     nroll trajectories of nstep controls
  output arguments (optional):
    state              (nroll x nstep x nstate)       nroll nstep states
    sensordata         (nroll x nstep x nsendordata)  nroll trajectories of nstep sensordata vectors
)";

// C-style rollout function, assumes all arguments are valid
// all input fields of d are initialised, contents at call time do not matter
// after returning, d will contain the last step of the last rollout
void _unsafe_rollout(const mjModel* m, mjData* d, int nroll, int nstep, unsigned int control_spec,
                     const mjtNum* state0, const mjtNum* warmstart0, const mjtNum* control,
                     mjtNum* state, mjtNum* sensordata) {
  // sizes
  int nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS);
  int ncontrol = mj_stateSize(m, control_spec);
  int nv = m->nv, nbody = m->nbody, neq = m->neq;
  int nsensordata = m->nsensordata;

  // clear user inputs if unspecified
  if (!(control_spec & mjSTATE_CTRL)) {
    mju_zero(d->ctrl, m->nu);
  }
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
    mju_zero(d->qfrc_applied, nv);
  }
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
    mju_zero(d->xfrc_applied, 6*nbody);
  }
  if (!(control_spec & mjSTATE_MOCAP_POS)) {
    for (int i = 0; i < nbody; i++) {
      int id = m->body_mocapid[i];
      if (id >= 0) mju_copy3(d->mocap_pos+3*id, m->body_pos+3*i);
    }
  }
  if (!(control_spec & mjSTATE_MOCAP_QUAT)) {
    for (int i = 0; i < nbody; i++) {
      int id = m->body_mocapid[i];
      if (id >= 0) mju_copy4(d->mocap_quat+4*id, m->body_quat+4*i);
    }
  }
  if (!(control_spec & mjSTATE_EQ_ACTIVE)) {
    for (int i = 0; i < neq; i++) {
      d->eq_active[i] = m->eq_active0[i];
    }
  }

  // loop over rollouts
  for (int r = 0; r < nroll; r++) {
    // set initial state
    mj_setState(m, d, state0 + r*nstate, mjSTATE_FULLPHYSICS);

    // set warmstart accelerations
    if (warmstart0) {
      mju_copy(d->qacc_warmstart, warmstart0 + r*nv, nv);
    } else {
      mju_zero(d->qacc_warmstart, nv);
    }

    // clear warning counters
    for (int i = 0; i < mjNWARNING; i++) {
      d->warning[i].number = 0;
    }

    // roll out trajectory
    for (int t = 0; t < nstep; t++) {
      // check for warnings
      bool nwarning = false;
      for (int i = 0; i < mjNWARNING; i++) {
        if (d->warning[i].number) {
          nwarning = true;
          break;
        }
      }

      // if any warnings, fill remaining outputs with current outputs, break
      if (nwarning) {
        for (; t < nstep; t++) {
          int step = r*nstep + t;
          if (state) {
            mj_getState(m, d, state + step*nstate, mjSTATE_FULLPHYSICS);
          }
          if (sensordata) {
            mju_copy(sensordata + step*nsensordata, d->sensordata, nsensordata);
          }
        }
        break;
      }

      int step = r*nstep + t;

      // controls
      if (control) {
        mj_setState(m, d, control + step*ncontrol, control_spec);
      }

      // step
      mj_step(m, d);

      // copy out new state
      if (state) {
        mj_getState(m, d, state + step*nstate, mjSTATE_FULLPHYSICS);
      }

      // copy out sensor values
      if (sensordata) {
        mju_copy(sensordata + step*nsensordata, d->sensordata, nsensordata);
      }
    }
  }
}

// NOLINTEND(whitespace/line_length)

struct UnsafeRolloutArgs_ {
  const mjModel* m;
  mjData* d;
  int nroll;
  int nstep;
  unsigned int control_spec;
  const mjtNum* state0;
  const mjtNum* warmstart0;
  const mjtNum* control;
  mjtNum* state;
  mjtNum* sensordata;
};
typedef struct UnsafeRolloutArgs_ UnsafeRolloutArgs;

void* _unsafe_rollout_with_arg(void* arg) {
  UnsafeRolloutArgs* a = static_cast<UnsafeRolloutArgs*>(arg);
  _unsafe_rollout(a->m, a->d, a->nroll, a->nstep, a->control_spec,
    a->state0, a->warmstart0, a->control, a->state, a->sensordata);
  return nullptr;
}

// Dispatch rollouts of multiple models through _unsafe_rollout
// Arguments have the same properties as _unsafe_rollout
void _unsafe_rollouts(const mjModel** m, mjData** d, int nmodel, int ndata, int* nroll, int* nstep, unsigned int* control_spec,
                      const mjtNum** state0, const mjtNum** warmstart0, const mjtNum** control,
                      mjtNum** state, mjtNum** sensordata,
                      int nthread) {

  if (nthread == 0) {
      for (int i = 0; i < nmodel; i++) {
          // check that some steps need to be taken, return if not
          if (nroll[i] < 1 || nstep[i] < 1) {
            return;
          }

        _unsafe_rollout(m[i], d[i], nroll[i], nstep[i], control_spec[i], 
          state0[i], warmstart0[i], control[i], state[i], sensordata[i]);
      }
  }
  else if (nmodel == ndata) {
    mjThreadPool* pool = mju_threadPoolCreate(nthread);

    mjTask tasks[nmodel];
    UnsafeRolloutArgs args[nmodel];
    for (int i = 0; i < nmodel; i++) {
      args[i].m = m[i];
      args[i].d = d[i];
      args[i].nroll = nroll[i];
      args[i].nstep = nstep[i];
      args[i].control_spec = control_spec[i];
      args[i].state0 = state0[i];
      args[i].warmstart0 = warmstart0[i];
      args[i].control = control[i];
      args[i].state = state[i];
      args[i].sensordata = sensordata[i];

      mju_defaultTask(&tasks[i]);
      tasks[i].func = &_unsafe_rollout_with_arg;
      tasks[i].args = &args[i];
      mju_threadPoolEnqueue(pool, &tasks[i]);
    }

    for (int i = 0; i < nmodel; i++) {
      mju_taskJoin(&tasks[i]);
    }

    mju_threadPoolDestroy(pool);
  }
  else { // nmodel != ndata, therefore one thread per ndata
    mjThreadPool* threads[ndata];
    for (int i = 0; i < ndata; i++) {
      threads[i] = mju_threadPoolCreate(1);
      // mju_bindThreadPool(d[i], threads[i]);
    }

    mjTask tasks[nmodel];
    UnsafeRolloutArgs args[nmodel];
    for (int i = 0; i < nmodel; i++) {
      int thread_index = i * nthread / nmodel;
      // std::cout << i << " " << thread_index << std::endl;

      args[i].m = m[i];
      args[i].d = d[thread_index];
      args[i].nroll = nroll[i];
      args[i].nstep = nstep[i];
      args[i].control_spec = control_spec[i];
      args[i].state0 = state0[i];
      args[i].warmstart0 = warmstart0[i];
      args[i].control = control[i];
      args[i].state = state[i];
      args[i].sensordata = sensordata[i];

      mju_defaultTask(&tasks[i]);
      tasks[i].func = &_unsafe_rollout_with_arg;
      tasks[i].args = &args[i];
      mju_threadPoolEnqueue(threads[thread_index], &tasks[i]);
    }

    for (int i = 0; i < nmodel; i++) {
      mju_taskJoin(&tasks[i]);
    }

    for (int i = 0; i < ndata; i++) {
      mju_threadPoolDestroy(threads[i]);
    }
  }
}

// check size of optional argument to rollout(), return raw pointer
mjtNum* get_array_ptr(std::optional<py::list> arg,
                      int i,
                      const char* name, int nroll, int nstep, int dim) {
  // if empty return nullptr
  if (!arg.has_value()) {
    return nullptr;
  }

  // get info
  // const PyCArray
  py::buffer_info info = (*arg)[i].cast<const py::array_t<mjtNum>>().request();

  // check size
  int expected_size = nroll * nstep * dim;
  if (info.size != expected_size) {
    std::ostringstream msg;
    msg << name << ".size should be " << expected_size << ", got " << info.size;
    throw py::value_error(msg.str());
  }
  return static_cast<mjtNum*>(info.ptr);
}


PYBIND11_MODULE(_rollout, pymodule) {
  namespace py = ::pybind11;
  // using PyCArray = py::array_t<mjtNum, py::array::c_style>;

  // roll out open loop trajectories from multiple initial states
  // get subsequent states and corresponding sensor values
  pymodule.def(
      "rollout",
      [](py::list m, py::list d,
         py::list nroll, py::list nstep, py::list control_spec,
         py::list state0,
         std::optional<py::list> warmstart0,
         std::optional<py::list> control,
         std::optional<py::list> state,
         std::optional<py::list> sensordata,
         std::optional<int> nthread
         ) {
        int num_threads = 0;
        if (nthread.has_value()) {
          num_threads = *nthread;
        }

        int nmodel = py::len(m);
        int ndata = py::len(d);
        const raw::MjModel* model_ptrs[nmodel];
        raw::MjData* data_ptrs[nmodel];
        unsigned int control_spec_ints[nmodel];
        int nroll_ints[nmodel];
        int nstep_ints[nmodel];
        const mjtNum* state0_ptrs[nmodel];
        const mjtNum* warmstart0_ptrs[nmodel];
        const mjtNum* control_ptrs[nmodel];
        mjtNum* state_ptrs[nmodel];
        mjtNum* sensordata_ptrs[nmodel];

        for (unsigned int i = 0; i < ndata; i++) {
          raw::MjData* data = d[i].cast<MjDataWrapper*>()->get();
          data_ptrs[i] = data;
        }

        for (unsigned int i = 0; i < nmodel; i++) {          
          // get sizes
          const raw::MjModel* model = m[i].cast<const MjModelWrapper*>()->get();

          control_spec_ints[i] = control_spec[i].cast<unsigned int>();
          nroll_ints[i] = nroll[i].cast<int>(); 
          nstep_ints[i] = nstep[i].cast<int>();
          int nstate = mj_stateSize(model, mjSTATE_FULLPHYSICS);
          int ncontrol = mj_stateSize(model, control_spec_ints[i]);

          // get raw pointers
          model_ptrs[i] = model;

          state0_ptrs[i] = get_array_ptr(state0, i, "state0", nroll_ints[i], 1, nstate);
          warmstart0_ptrs[i] = get_array_ptr(warmstart0, i, "warmstart0", nroll_ints[i],
                                                 1, model->nv);
          control_ptrs[i] = get_array_ptr(control, i, "control", nroll_ints[i],
                                              nstep_ints[i], ncontrol);
          state_ptrs[i] = get_array_ptr(state, i, "state", nroll_ints[i], nstep_ints[i], nstate);
          sensordata_ptrs[i] = get_array_ptr(sensordata, i, "sensordata", nroll_ints[i],
                                                 nstep_ints[i], model->nsensordata);
        }

        // perform rollouts
        {
          // release the GIL
          py::gil_scoped_release no_gil;

          // call unsafe rollout function
          InterceptMjErrors(_unsafe_rollouts)(
              model_ptrs, data_ptrs, nmodel, ndata, nroll_ints, nstep_ints, control_spec_ints, state0_ptrs,
              warmstart0_ptrs, control_ptrs, state_ptrs, sensordata_ptrs, num_threads);
        }
      },
      py::arg("model"),
      py::arg("data"),
      py::arg("nroll"),
      py::arg("nstep"),
      py::arg("control_spec"),
      py::arg("state0"),
      py::arg("warmstart0") = py::none(),
      py::arg("control")    = py::none(),
      py::arg("state")      = py::none(),
      py::arg("sensordata") = py::none(),
      py::arg("nthread") = py::none(),
      py::doc(rollout_doc)
  );
}

}  // namespace

}  // namespace mujoco::python

