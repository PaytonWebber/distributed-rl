cmake_minimum_required(VERSION 3.10)
project(actors)

add_subdirectory(
    ../common        
    common_build     
)

include_directories(include)

add_executable(actors
    src/main.cpp
    src/actor.cpp
)

target_link_libraries(actors PRIVATE common)
