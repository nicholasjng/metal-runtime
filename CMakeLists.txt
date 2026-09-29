cmake_minimum_required(VERSION 3.24)
project(metal_runtime LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR
    "metal-runtime targets Apple platforms only: it builds against Metal, "
    "which exists only on macOS/iOS.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Developer builds copy their compilation database to the repository root;
# isolated package builds must not.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
option(METAL_RUNTIME_SYNC_COMPILE_COMMANDS
  "Copy compile_commands.json to the source root" OFF)
if(PROJECT_IS_TOP_LEVEL AND METAL_RUNTIME_SYNC_COMPILE_COMMANDS)
  add_custom_target(
    sync_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    COMMENT "Syncing compile_commands.json to source root"
    VERBATIM)
endif()

# Apple's metal-cpp, header-only. Some releases nest under metal-cpp/.
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

add_library(metal-cpp INTERFACE)
target_include_directories(metal-cpp INTERFACE "${METAL_CPP_INCLUDE}")
target_compile_definitions(metal-cpp INTERFACE METALCPP_SYMBOL_VISIBILITY_HIDDEN)
target_link_libraries(metal-cpp INTERFACE
  "-framework Foundation" "-framework Metal" "-framework QuartzCore")

# One shared library: runtime, metal-cpp implementation, and the C API. The
# Python extension links against it, so a process hosting both (e.g. an XLA
# FFI handler) has one device, one queue, one cache. Ships in the wheel.
add_library(metal_runtime SHARED
  src/runtime.cpp
  src/metal.cpp
  src/buffer.cpp
  src/dtype.cpp
  src/library.cpp
  src/dispatch.cpp
  src/c_api.cpp)
target_include_directories(metal_runtime PUBLIC src)   # so #include "runtime.h" resolves
target_link_libraries(metal_runtime PRIVATE metal-cpp)
set_target_properties(metal_runtime PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  INSTALL_NAME_DIR "@rpath")
install(TARGETS metal_runtime LIBRARY DESTINATION metal_runtime)
install(FILES src/c_api.h DESTINATION metal_runtime/include)

# A plain `cmake -S . -B build` without nanobind still builds metal_runtime.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module)

set(METAL_RUNTIME_HAVE_NANOBIND FALSE)
if(Python_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nanobind_cmake_dir_status
    ERROR_QUIET
  )
  if(_nanobind_cmake_dir_status EQUAL 0)
    find_package(nanobind CONFIG QUIET)
    if(nanobind_FOUND)
      set(METAL_RUNTIME_HAVE_NANOBIND TRUE)
    endif()
  endif()
endif()

if(METAL_RUNTIME_HAVE_NANOBIND)
  nanobind_add_module(
    _core
    NB_STATIC
    FREE_THREADED
    src/bindings.cpp
  )
  target_link_libraries(_core PRIVATE metal_runtime)
  set_target_properties(_core PROPERTIES INSTALL_RPATH "@loader_path")

  install(TARGETS _core LIBRARY DESTINATION metal_runtime)

  set(_METAL_RUNTIME_CORE_STUB "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi")
  nanobind_add_stub(
    _core_stub
    MODULE _core
    OUTPUT ${_METAL_RUNTIME_CORE_STUB}
    PYTHON_PATH $<TARGET_FILE_DIR:_core>
    DEPENDS _core
  )

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/metal_runtime/_core.pyi
    DESTINATION metal_runtime)

  # Format generated stubs when Ruff is available.
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m ruff --version
    RESULT_VARIABLE _ruff_available
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_ruff_available EQUAL 0)
    add_custom_command(
      TARGET _core_stub POST_BUILD
      COMMAND ${Python_EXECUTABLE} -m ruff check --fix --exit-zero --quiet
        ${_METAL_RUNTIME_CORE_STUB}
      COMMAND ${Python_EXECUTABLE} -m ruff format --quiet
        ${_METAL_RUNTIME_CORE_STUB}
      COMMENT "ruff fix + format _core.pyi"
      VERBATIM
    )
  endif()

  add_custom_target(
    update_metal_runtime_core_stub
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${_METAL_RUNTIME_CORE_STUB}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/metal_runtime/_core.pyi
    DEPENDS _core_stub
    COMMENT "Updating checked-in src/metal_runtime/_core.pyi"
    VERBATIM)

  add_custom_target(
    check_metal_runtime_core_stub
    COMMAND ${CMAKE_COMMAND} -E compare_files
      ${_METAL_RUNTIME_CORE_STUB}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/metal_runtime/_core.pyi
    DEPENDS _core_stub
    COMMENT "Checking checked-in src/metal_runtime/_core.pyi"
    VERBATIM)
else()
  message(STATUS "nanobind not found for ${Python_EXECUTABLE} -- skipping the "
                  "_core Python extension target (metal_runtime still builds). "
                  "Use `uv sync` to build the full Python package.")
endif()
