#!/bin/bash -xe
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ -z ${VIRTUAL_ENV} ]] && [[ -z ${CONDA_DEFAULT_ENV} ]]; then
  echo "This script must be run from within a Python virtual environment"
  exit 1
fi


package_dir="."








python -m pip install --upgrade --require-hashes \
    -r ${package_dir}/make_sdist_requirements.txt
# pushd ${tmp_dir}
# cp -r "${package_dir}"/* .

# Generate header files.
old_pythonpath="${PYTHONPATH}"
if [[ "$(uname)" == CYGWIN* || "$(uname)" == MINGW* ]]; then
  export PYTHONPATH="${old_pythonpath};${package_dir}/mujoco/python/.."
else
  export PYTHONPATH="${old_pythonpath}:${package_dir}/mujoco/python/.."
fi
python "${package_dir}"/mujoco/codegen/generate_enum_traits.py > \
    mujoco/enum_traits.h
python "${package_dir}"/mujoco/codegen/generate_function_traits.py > \
    mujoco/function_traits.h
python "${package_dir}"/mujoco/codegen/generate_spec_bindings.py > \
    mujoco/specs.cc.inc
export PYTHONPATH="${old_pythonpath}"

# Copy over the LICENSE file.
cp "${package_dir}"/../LICENSE .

# Copy over CMake scripts.
mkdir -p mujoco/cmake
cp "${package_dir}"/../cmake/*.cmake mujoco/cmake

# Copy over Simulate source code.
cp -r "${package_dir}"/../simulate mujoco

if [ -d "mujoco/include" ] 
then
    rm -r mujoco/include
fi

if [ -d "mujoco/plugin" ] 
then
    rm -r mujoco/plugin
fi

python -m pip install --verbose -e .
