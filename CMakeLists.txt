cmake_minimum_required(VERSION 3.26)

project(wmfs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CapnProto CONFIG REQUIRED)
find_package(Threads REQUIRED)

option(WMFS_BUILD_PYTHON_RUNTIME "Build the Python runtime extension" ON)
option(WMFS_BUILD_REFERENCE_WORKER "Build the C++ reference plugin worker" OFF)

set(CAPNPC_SRC_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/wmfs/schemas")
set(CAPNPC_IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/wmfs/schemas")
capnp_generate_cpp(
  WMFS_CAPNP_SOURCES
  WMFS_CAPNP_HEADERS
  wmfs/schemas/wmfs/tensor.capnp
  wmfs/schemas/wmfs/runtime.capnp
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
endif()

if(WMFS_BUILD_REFERENCE_WORKER)
  find_package(Torch CONFIG REQUIRED)

  set(CAPNPC_SRC_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/schemas")
  set(
    CAPNPC_IMPORT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/wmfs/schemas"
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
      "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_link_libraries(
    wmfs-reference-worker
    PRIVATE
      CapnProto::capnp-rpc
      CapnProto::kj-async
      Threads::Threads
      ${TORCH_LIBRARIES}
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
