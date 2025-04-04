cmake_minimum_required(VERSION 3.10)
project(common_lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "libs/libtorch")

find_package(Torch REQUIRED)
include_directories(include src)

add_library(common STATIC
  src/az_net.cpp
  src/residual_block.cpp
)

target_include_directories(common
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(common PUBLIC "${TORCH_LIBRARIES}")
