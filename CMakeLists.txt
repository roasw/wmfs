cmake_minimum_required(VERSION 3.26)

project(wmfs LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CapnProto CONFIG REQUIRED)
find_package(Threads REQUIRED)

option(WMFS_BUILD_PYTHON_RUNTIME "Build the Python runtime extension" ON)
option(WMFS_BUILD_REFERENCE_WORKER "Build the C++ reference plugin worker" OFF)
set(
  WMFS_BUNDLED_PLUGINS
  ""
  CACHE STRING
  "Semicolon-separated plugins to compile into the in-process backend"
)

if(BUILD_TESTING)
  add_executable(wmfs-unique-fd-test tests/unique_fd_test.cpp)
  target_include_directories(
    wmfs-unique-fd-test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  )
  add_test(NAME wmfs-unique-fd COMMAND wmfs-unique-fd-test)
endif()

set(WMFS_BUNDLE_REFERENCE OFF)
foreach(plugin IN LISTS WMFS_BUNDLED_PLUGINS)
  if(plugin STREQUAL "reference")
    set(WMFS_BUNDLE_REFERENCE ON)
  else()
    message(FATAL_ERROR "Unknown bundled plugin: ${plugin}")
  endif()
endforeach()
if(WMFS_BUNDLE_REFERENCE AND NOT WMFS_BUILD_PYTHON_RUNTIME)
  message(FATAL_ERROR "Bundled plugins require the Python runtime")
endif()

if(WMFS_BUILD_REFERENCE_WORKER OR WMFS_BUNDLE_REFERENCE)
  find_package(Torch CONFIG REQUIRED)
  add_library(wmfs-reference-kernels STATIC src/reference_kernels.cpp)
  set_target_properties(
    wmfs-reference-kernels
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(
    wmfs-reference-kernels
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  )
  target_link_libraries(wmfs-reference-kernels PUBLIC ${TORCH_LIBRARIES})
  target_compile_options(wmfs-reference-kernels PRIVATE ${TORCH_CXX_FLAGS})
endif()

set(
  WMFS_SCHEMA_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/packages/wmfs-plugin/wmfs_plugin/schemas"
)
set(CAPNPC_SRC_PREFIX "${WMFS_SCHEMA_ROOT}")
set(CAPNPC_IMPORT_DIRS "${WMFS_SCHEMA_ROOT}")
capnp_generate_cpp(
  WMFS_CAPNP_SOURCES
  WMFS_CAPNP_HEADERS
  packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/tensor.capnp
  packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/runtime.capnp
)

if(WMFS_BUILD_PYTHON_RUNTIME)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    _native
    NB_STATIC
    src/native_module.cpp
    src/native_session.cpp
    ${WMFS_CAPNP_SOURCES}
  )

  target_include_directories(
    _native
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
      "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_link_libraries(_native PRIVATE CapnProto::capnp-rpc CapnProto::kj-async)

  install(TARGETS _native LIBRARY DESTINATION wmfs)

  if(WMFS_BUNDLE_REFERENCE)
    nanobind_add_module(
      _bundled
      NB_STATIC
      src/bundled_module.cpp
      src/reference_torch_ops.cpp
    )
    target_link_libraries(_bundled PRIVATE wmfs-reference-kernels)
    target_compile_options(_bundled PRIVATE ${TORCH_CXX_FLAGS})
    set_target_properties(_bundled PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS _bundled LIBRARY DESTINATION wmfs)
  endif()
endif()

if(WMFS_BUILD_REFERENCE_WORKER)
  set(CAPNPC_SRC_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/schemas")
  set(
    CAPNPC_IMPORT_DIRS
    "${WMFS_SCHEMA_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/schemas"
  )
  capnp_generate_cpp(
    REFERENCE_CAPNP_SOURCES
    REFERENCE_CAPNP_HEADERS
    plugins/reference/schemas/wmfs-reference/reference.capnp
  )

  add_executable(
    wmfs-reference-worker
    src/reference_mapped_buffers.cpp
    src/reference_worker.cpp
    ${WMFS_CAPNP_SOURCES}
    ${REFERENCE_CAPNP_SOURCES}
  )
  target_include_directories(
    wmfs-reference-worker
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/generated"
      "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_link_libraries(
    wmfs-reference-worker
    PRIVATE
      CapnProto::capnp-rpc
      CapnProto::kj-async
      Threads::Threads
      wmfs-reference-kernels
  )
  target_compile_options(wmfs-reference-worker PRIVATE ${TORCH_CXX_FLAGS})
  set_target_properties(
    wmfs-reference-worker
    PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  install(TARGETS wmfs-reference-worker RUNTIME DESTINATION bin)
  install(
    FILES plugins/reference/plugin.toml
    DESTINATION share/wmfs/plugins/reference
  )
  install(
    FILES plugins/reference/schemas/wmfs-reference/reference.capnp
    DESTINATION share/wmfs/plugins/reference/schemas/wmfs-reference
  )
endif()
