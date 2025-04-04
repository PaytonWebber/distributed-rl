cmake_minimum_required(VERSION 3.10)
project(learner)

include(FetchContent)

FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.10.0
)

FetchContent_MakeAvailable(cppzmq)

FetchContent_Declare(
  zeromq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.5
)
set(ENABLE_TESTS OFF CACHE BOOL "Disable ZeroMQ tests" FORCE)

FetchContent_MakeAvailable(zeromq)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_subdirectory(
    ../common        
    common_build     
)

include_directories(include)

add_executable(learner
    src/main.cpp
    src/learner.cpp
)

target_link_libraries(learner
  PRIVATE
    common
    libzmq
    cppzmq
    nlohmann_json::nlohmann_json
)
