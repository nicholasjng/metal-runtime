cmake_minimum_required(VERSION 3.24)
project(metal_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # build/compile_commands.json for clangd

# Apple's official metal-cpp, fetched at configure time (replaces the old
# download_metal_cpp.py). Header-only: Foundation / Metal / QuartzCore, no MPS.
# Bump the version by changing the URL; the archive nests under metal-cpp/ on
# some releases, so handle either layout.
include(FetchContent)
FetchContent_Declare(
  metalcpp
  URL https://developer.apple.com/metal/cpp/files/metal-cpp_26.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metalcpp)
if(EXISTS "${metalcpp_SOURCE_DIR}/Metal/Metal.hpp")
  set(METAL_CPP_INCLUDE "${metalcpp_SOURCE_DIR}")
else()
  set(METAL_CPP_INCLUDE "${metalcpp_SOURCE_DIR}/metal-cpp")
endif()

# metal-cpp is header-only; the implementation is generated in exactly one TU via
# the NS_/MTL_/CA_PRIVATE_IMPLEMENTATION macros (see src/metal_probe.cpp).
add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE "${METAL_CPP_INCLUDE}")
target_link_libraries(metal-cpp INTERFACE
  "-framework Foundation" "-framework Metal" "-framework QuartzCore")

# Smoke test: print the system default GPU. Proves the metal-cpp fetch + the
# Objective-C runtime bridge work. The runtime core + nanobind bindings build on
# top of the metal-cpp target — see ROADMAP.md.
add_executable(metal_probe src/metal_probe.cpp)
target_link_libraries(metal_probe PRIVATE metal-cpp)
