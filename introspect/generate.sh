#!/bin/bash
set -e

export PYTHONPATH="$(pwd)/.."
export MUJOCO_PATH="$(pwd)/.."
MUJOCO_H_JSON=$(mktemp /tmp/generate-mujoco-py.XXXXXX)

clang -Xclang -ast-dump=json -fsyntax-only -fparse-all-comments -x c -I "$MUJOCO_PATH/include" "$MUJOCO_PATH/include/mujoco/mujoco.h" > "$MUJOCO_H_JSON"

python codegen/generate_enums.py     --json_path "$MUJOCO_H_JSON" > "$MUJOCO_PATH/introspect/enums.py"
python codegen/generate_structs.py   --json_path "$MUJOCO_H_JSON" > "$MUJOCO_PATH/introspect/structs.py"
python codegen/generate_functions.py --json_path "$MUJOCO_H_JSON" --header_path "$MUJOCO_PATH/include/mujoco/mujoco.h" > "$MUJOCO_PATH/introspect/functions.py"

rm "$MUJOCO_H_JSON"
