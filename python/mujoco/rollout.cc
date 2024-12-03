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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <absl/base/attributes.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;

// Copied from https://github.com/google-deepmind/mujoco_mpc/blob/main/mjpc/threadpool.h
// ThreadPool class
class ThreadPool {
 public:
  // constructor
  explicit ThreadPool(int num_threads);
  // destructor
  ~ThreadPool();
  int NumThreads() const { return threads_.size(); }
  // returns an ID between 0 and NumThreads() - 1. must be called within
  // worker thread (returns -1 if not).
  static int WorkerId() { return worker_id_; }
  // ----- methods ----- //
  // set task for threadpool
  void Schedule(std::function<void()> task);
  // return number of tasks completed
  std::uint64_t GetCount() { return ctr_; }
  // reset count to zero
  void ResetCount() { ctr_ = 0; }
  // wait for count, then return
  void WaitCount(int value) {
    std::unique_lock<std::mutex> lock(m_);
    cv_ext_.wait(lock, [&]() { return this->GetCount() >= value; });
  }
 private:
  // ----- methods ----- //
  // execute task with available thread
  void WorkerThread(int i);
  ABSL_CONST_INIT static thread_local int worker_id_;
  // ----- members ----- //
  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_in_;
  std::condition_variable cv_ext_;
  std::queue<std::function<void()>> queue_;
  std::uint64_t ctr_;
};

// Copied from https://github.com/google-deepmind/mujoco_mpc/blob/main/mjpc/threadpool.cc
ABSL_CONST_INIT thread_local int ThreadPool::worker_id_ = -1;
// ThreadPool constructor
ThreadPool::ThreadPool(int num_threads) : ctr_(0) {
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(std::thread(&ThreadPool::WorkerThread, this, i));
  }
}
// ThreadPool destructor
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_);
    for (int i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);
    }
    cv_in_.notify_all();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}
// ThreadPool scheduler
void ThreadPool::Schedule(std::function<void()> task) {
  std::unique_lock<std::mutex> lock(m_);
  queue_.push(std::move(task));
  cv_in_.notify_one();
}
// ThreadPool worker
void ThreadPool::WorkerThread(int i) {
  worker_id_ = i;
  while (true) {
    auto task = [&]() {
      std::unique_lock<std::mutex> lock(m_);
      cv_in_.wait(lock, [&]() { return !queue_.empty(); });
      std::function<void()> task = std::move(queue_.front());
      queue_.pop();
      cv_in_.notify_one();
      return task;
    }();
    if (task == nullptr) {
      {
        std::unique_lock<std::mutex> lock(m_);
        ++ctr_;
        cv_ext_.notify_one();
      }
      break;
    }
    task();
    {
      std::unique_lock<std::mutex> lock(m_);
      ++ctr_;
      cv_ext_.notify_one();
    }
  }
}

// NOLINTBEGIN(whitespace/line_length)

const auto rollout_doc = R"(
Roll out open-loop trajectories from initial states, get resulting states and sensor values.

  input arguments (required):
    model              list of MjModel instances of length nroll
    data               associated instance of MjData
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
void _unsafe_rollout(std::vector<const mjModel*>& m, mjData* d, int start_roll, int end_roll, int nstep, unsigned int control_spec,
                     const mjtNum* state0, const mjtNum* warmstart0, const mjtNum* control,
                     mjtNum* state, mjtNum* sensordata) {
  // sizes
  int nstate = mj_stateSize(m[0], mjSTATE_FULLPHYSICS);
  int ncontrol = mj_stateSize(m[0], control_spec);
  int nv = m[0]->nv, nbody = m[0]->nbody, neq = m[0]->neq;
  int nsensordata = m[0]->nsensordata;

  // clear user inputs if unspecified
  if (!(control_spec & mjSTATE_CTRL)) {
    mju_zero(d->ctrl, m[0]->nu);
  }
  if (!(control_spec & mjSTATE_QFRC_APPLIED)) {
    mju_zero(d->qfrc_applied, nv);
  }
  if (!(control_spec & mjSTATE_XFRC_APPLIED)) {
    mju_zero(d->xfrc_applied, 6*nbody);
  }

  // loop over rollouts
  for (int r = start_roll; r < end_roll; r++) {
    // clear user inputs if unspecified
    if (!(control_spec & mjSTATE_MOCAP_POS)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy3(d->mocap_pos+3*id, m[r]->body_pos+3*i);
      }
    }
    if (!(control_spec & mjSTATE_MOCAP_QUAT)) {
      for (int i = 0; i < nbody; i++) {
        int id = m[r]->body_mocapid[i];
        if (id >= 0) mju_copy4(d->mocap_quat+4*id, m[r]->body_quat+4*i);
      }
    }
    if (!(control_spec & mjSTATE_EQ_ACTIVE)) {
      for (int i = 0; i < neq; i++) {
        d->eq_active[i] = m[r]->eq_active0[i];
      }
    }

    // set initial state
    mj_setState(m[r], d, state0 + r*nstate, mjSTATE_FULLPHYSICS);

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
            mj_getState(m[r], d, state + step*nstate, mjSTATE_FULLPHYSICS);
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
        mj_setState(m[r], d, control + step*ncontrol, control_spec);
      }

      // step
      mj_step(m[r], d);

      // copy out new state
      if (state) {
        mj_getState(m[r], d, state + step*nstate, mjSTATE_FULLPHYSICS);
      }

      // copy out sensor values
      if (sensordata) {
        mju_copy(sensordata + step*nsensordata, d->sensordata, nsensordata);
      }
    }
  }
}

// C-style threaded version of _unsafe_rollout
static ThreadPool* pool = nullptr;
void _unsafe_rollout_threaded(std::vector<const mjModel*>& m, std::vector<mjData*>& d,
                              int nroll, int nstep, unsigned int control_spec,
                              const mjtNum* state0, const mjtNum* warmstart0, const mjtNum* control,
                              mjtNum* state, mjtNum* sensordata,
                              int nthread, int chunk_size) {
  int nfulljobs = nroll / chunk_size;
  int chunk_remainder = nroll % chunk_size;
  int njobs = nfulljobs;
  if (chunk_remainder > 0) njobs++;

  if (pool == nullptr) {
    pool = new ThreadPool(nthread);
  }
  else if (pool->NumThreads() != nthread) {
    delete pool; // TODO make sure pool is shutdown correctly
    pool = new ThreadPool(nthread);
  } else {
    pool->ResetCount();
  }

  for (int j = 0; j < nfulljobs; j++) {
    auto task = [=, &m, &d](void) {
      int id = pool->WorkerId();
      _unsafe_rollout(m, d[id], j*chunk_size, (j+1)*chunk_size,
        nstep, control_spec, state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }

  if (chunk_remainder > 0) {
    auto task = [=, &m, &d](void) {
      _unsafe_rollout(m, d[pool->WorkerId()], nfulljobs*chunk_size, nfulljobs*chunk_size+chunk_remainder,
        nstep, control_spec, state0, warmstart0, control, state, sensordata);
    };
    pool->Schedule(task);
  }

  pool->WaitCount(njobs);
}

// NOLINTEND(whitespace/line_length)

// check size of optional argument to rollout(), return raw pointer
mjtNum* get_array_ptr(std::optional<const py::array_t<mjtNum>> arg,
                      const char* name, int nroll, int nstep, int dim) {
  // if empty return nullptr
  if (!arg.has_value()) {
    return nullptr;
  }

  // get info
  py::buffer_info info = arg->request();

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
  using PyCArray = py::array_t<mjtNum, py::array::c_style>;

  // roll out open loop trajectories from multiple initial states
  // get subsequent states and corresponding sensor values
  pymodule.def(
      "rollout",
      [](py::list m, py::list d,
         int nstep, unsigned int control_spec,
         const PyCArray state0,
         std::optional<const PyCArray> warmstart0,
         std::optional<const PyCArray> control,
         std::optional<const PyCArray> state,
         std::optional<const PyCArray> sensordata
         ) {
        // get raw pointers
        int nroll = state0.shape(0);
        std::vector<const raw::MjModel*> model_ptrs(nroll);
        for (int r = 0; r < nroll; r++) {
          model_ptrs[r] = m[r].cast<const MjModelWrapper*>()->get();
        }

        int nthread = py::len(d);
        std::vector<raw::MjData*> data_ptrs(nthread);
        for (int t = 0; t < nthread; t++) {
          data_ptrs[t] = d[t].cast<MjDataWrapper*>()->get();
        }

        // check that some steps need to be taken, return if not
        if (nstep < 1) {
          return;
        }

        // get sizes
        int nstate = mj_stateSize(model_ptrs[0], mjSTATE_FULLPHYSICS);
        int ncontrol = mj_stateSize(model_ptrs[0], control_spec);

        mjtNum* state0_ptr = get_array_ptr(state0, "state0", nroll, 1, nstate);
        mjtNum* warmstart0_ptr = get_array_ptr(warmstart0, "warmstart0", nroll,
                                               1, model_ptrs[0]->nv);
        mjtNum* control_ptr = get_array_ptr(control, "control", nroll,
                                            nstep, ncontrol);
        mjtNum* state_ptr = get_array_ptr(state, "state", nroll, nstep, nstate);
        mjtNum* sensordata_ptr = get_array_ptr(sensordata, "sensordata", nroll,
                                               nstep, model_ptrs[0]->nsensordata);

        // perform rollouts
        {
          // release the GIL
          py::gil_scoped_release no_gil;

          // call unsafe rollout function
          if (nthread > 1 && nroll > 1) {
            int chunk_size = std::max(1, nroll / (10 * nthread));
            InterceptMjErrors(_unsafe_rollout_threaded)(
                model_ptrs, data_ptrs, nroll, nstep, control_spec, state0_ptr,
                warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr,
                nthread, chunk_size);
          }
          else {
            InterceptMjErrors(_unsafe_rollout)(
                model_ptrs, data_ptrs[0], 0, nroll, nstep, control_spec, state0_ptr,
                warmstart0_ptr, control_ptr, state_ptr, sensordata_ptr);
          }
        }
      },
      py::arg("model"),
      py::arg("data"),
      py::arg("nstep"),
      py::arg("control_spec"),
      py::arg("state0"),
      py::arg("warmstart0") = py::none(),
      py::arg("control")    = py::none(),
      py::arg("state")      = py::none(),
      py::arg("sensordata") = py::none(),
      py::doc(rollout_doc)
  );
}

}  // namespace

}  // namespace mujoco::python

