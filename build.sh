#!/bin/bash

rm -rf build

TORCH_PREFIX=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')
meson setup -Dcmake_prefix_path=$TORCH_PREFIX build
