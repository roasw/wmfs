cmake_minimum_required(VERSION 3.26)

project(wmfs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
find_package(CapnProto CONFIG REQUIRED)

set(CAPNPC_SRC_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/wmfs/schemas")
set(CAPNPC_IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/wmfs/schemas")
capnp_generate_cpp(
  WMFS_CAPNP_SOURCES
  WMFS_CAPNP_HEADERS
  wmfs/schemas/wmfs/tensor.capnp
  wmfs/schemas/wmfs/runtime.capnp
)

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
