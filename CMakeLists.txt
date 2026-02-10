cmake_minimum_required(VERSION 3.10)
project(NirithyAii)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/core/*.cpp")

# Create a library
add_library(nirithyaii_core ${SOURCES})

# Verification executable
add_executable(basic_tensor examples/basic_tensor.cpp)
target_link_libraries(basic_tensor nirithyaii_core)
