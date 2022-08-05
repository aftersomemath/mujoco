// Copyright 2021 DeepMind Technologies Limited
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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "glfw_dispatch.h"
#include "simulate.h"
#include "array_safety.h"

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;
} // namespace

//------------------------------------------ main --------------------------------------------------

// These functions can be used to destroy/create objects
// relevant to control tasks on top of the base MuJoCo physics
// For instance here an object could be created and a member
// function could be set to the mjcb_control callback
void preload(const mjModel*, mjData*) {}
void load(const mjModel*, mjData*) {}

// run event loop
int main(int argc, const char** argv) {
  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER!=mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>();

  // init GLFW
  mj::Glfw().glfwInit();

  const char* filename = nullptr;
  if (argc >  1) {
    filename = argv[1];
  }

  // start physics thread
  std::thread physicsthreadhandle = std::thread(
    [&](){ mj::PhysicsThread(*sim.get(), filename, &preload, &load); }
  );

  // start simulation UI loop (blocking call)
  sim->renderloop();
  physicsthreadhandle.join();

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  mj::Glfw().glfwTerminate();
#endif

  return 0;
}
