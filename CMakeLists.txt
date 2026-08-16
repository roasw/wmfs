cmake_minimum_required(VERSION 3.26)

if(NOT WMFS_VERSION)
  if(DEFINED ENV{WMFS_GIT_VERSION})
    set(WMFS_VERSION "$ENV{WMFS_GIT_VERSION}")
  else()
    find_package(Git QUIET)
    if(Git_FOUND)
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE WMFS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE WMFS_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(WMFS_VERSION AND WMFS_GIT_STATUS)
        string(APPEND WMFS_VERSION "-dirty")
      endif()
    endif()
  endif()
endif()
if(NOT WMFS_VERSION)
  set(WMFS_VERSION "unknown")
endif()

project(wmfs LANGUAGES CXX)
message(STATUS "WMFS version: ${WMFS_VERSION}")

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CapnProto CONFIG REQUIRED)
find_package(Threads REQUIRED)

option(WMFS_BUILD_PYTHON_RUNTIME "Build the Python runtime extension" ON)
option(WMFS_BUILD_REFERENCE_WORKER "Build the C++ reference plugin worker" OFF)
option(WMFS_BUILD_DOCUMENTATION "Build the unified API documentation" OFF)
set(
  WMFS_BUNDLED_PLUGINS
  ""
  CACHE STRING
  "Semicolon-separated plugins to compile into the in-process backend"
)

if(WMFS_BUILD_DOCUMENTATION)
  find_package(Doxygen REQUIRED)
  find_program(WMFS_BASH_EXECUTABLE bash REQUIRED)
  find_program(WMFS_SPHINX_EXECUTABLE sphinx-build REQUIRED)
  add_custom_target(
    doc
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "WMFS_CMAKE_COMMAND=${CMAKE_COMMAND}"
      "WMFS_DOXYGEN_EXECUTABLE=${DOXYGEN_EXECUTABLE}"
      "WMFS_GIT_VERSION=${WMFS_VERSION}"
      "WMFS_SPHINX_EXECUTABLE=${WMFS_SPHINX_EXECUTABLE}"
      "${WMFS_BASH_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/docs/build.sh"
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/docs"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/docs/html/index.html"
    COMMENT "Building unified Python and C++ documentation"
    VERBATIM
  )
endif()

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
add_library(wmfs-protocol STATIC ${WMFS_CAPNP_SOURCES})
add_library(wmfs::protocol ALIAS wmfs-protocol)
set_target_properties(wmfs-protocol PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  wmfs-protocol
  PUBLIC "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(wmfs-protocol PUBLIC CapnProto::capnp-rpc)

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
    wmfs-native-runtime
    NB_STATIC
    src/native_module.cpp
    src/native_session.cpp
  )
  add_library(wmfs::native-runtime ALIAS wmfs-native-runtime)
  set_target_properties(wmfs-native-runtime PROPERTIES OUTPUT_NAME _native)

  target_include_directories(
    wmfs-native-runtime
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  )
  target_link_libraries(
    wmfs-native-runtime
    PRIVATE wmfs::protocol CapnProto::kj-async
  )

  install(
    TARGETS wmfs-native-runtime
    LIBRARY DESTINATION wmfs
    COMPONENT runtime
  )

  if(WMFS_BUNDLE_REFERENCE)
    nanobind_add_module(
      wmfs-bundled-reference
      NB_STATIC
      src/bundled_module.cpp
      src/reference_torch_ops.cpp
    )
    add_library(wmfs::bundled-reference ALIAS wmfs-bundled-reference)
    target_link_libraries(wmfs-bundled-reference PRIVATE wmfs-reference-kernels)
    target_compile_options(wmfs-bundled-reference PRIVATE ${TORCH_CXX_FLAGS})
    set_target_properties(
      wmfs-bundled-reference
      PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        OUTPUT_NAME _bundled
    )
    install(
      TARGETS wmfs-bundled-reference
      LIBRARY DESTINATION wmfs
      COMPONENT runtime
    )
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
      wmfs::protocol
      wmfs-reference-kernels
  )
  target_compile_options(wmfs-reference-worker PRIVATE ${TORCH_CXX_FLAGS})
  set_target_properties(
    wmfs-reference-worker
    PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  if(BUILD_TESTING)
    add_executable(
      wmfs-reference-mapped-buffers-test
      tests/reference_mapped_buffers_test.cpp
      src/reference_mapped_buffers.cpp
    )
    target_include_directories(
      wmfs-reference-mapped-buffers-test
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
        "${CMAKE_CURRENT_BINARY_DIR}"
    )
    target_link_libraries(
      wmfs-reference-mapped-buffers-test
      PRIVATE wmfs::protocol Threads::Threads ${TORCH_LIBRARIES}
    )
    target_compile_options(
      wmfs-reference-mapped-buffers-test
      PRIVATE ${TORCH_CXX_FLAGS}
    )
    set_target_properties(
      wmfs-reference-mapped-buffers-test
      PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    foreach(operation IN ITEMS retire finish close)
      add_test(
        NAME wmfs-reference-mapped-buffers-${operation}
        COMMAND wmfs-reference-mapped-buffers-test ${operation}
      )
    endforeach()
  endif()

  install(
    TARGETS wmfs-reference-worker
    RUNTIME DESTINATION bin
    COMPONENT reference-worker
  )
  install(
    FILES plugins/reference/plugin.toml
    DESTINATION share/wmfs/plugins/reference
    COMPONENT reference-worker
  )
  install(
    FILES plugins/reference/schemas/wmfs-reference/reference.capnp
    DESTINATION share/wmfs/plugins/reference/schemas/wmfs-reference
    COMPONENT reference-worker
  )
endif()
