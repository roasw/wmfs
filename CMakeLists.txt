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

project(wmfs LANGUAGES C CXX)
message(STATUS "WMFS version: ${WMFS_VERSION}")

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(wmfs-ring STATIC src/ring.cpp)
add_library(wmfs::ring ALIAS wmfs-ring)
target_include_directories(
  wmfs-ring
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

add_library(wmfs-control STATIC src/control.cpp)
add_library(wmfs::control ALIAS wmfs-control)
target_include_directories(wmfs-control PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
set_target_properties(
  wmfs-control
  PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

add_library(wmfs-log STATIC src/log.cpp)
add_library(wmfs::log ALIAS wmfs-log)
target_include_directories(wmfs-log PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
set_target_properties(wmfs-log PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

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
  find_program(WMFS_D2_EXECUTABLE d2 REQUIRED)
  find_program(WMFS_SPHINX_EXECUTABLE sphinx-build REQUIRED)
  add_custom_target(
    doc
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "WMFS_CMAKE_COMMAND=${CMAKE_COMMAND}"
      "WMFS_DOXYGEN_EXECUTABLE=${DOXYGEN_EXECUTABLE}"
      "WMFS_D2_EXECUTABLE=${WMFS_D2_EXECUTABLE}"
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
  add_executable(wmfs-control-layout-c-test tests/cpp/control_layout_test.c)
  target_include_directories(wmfs-control-layout-c-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  set_target_properties(wmfs-control-layout-c-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  add_test(NAME wmfs-control-layout-c COMMAND wmfs-control-layout-c-test)

  add_executable(wmfs-control-test tests/cpp/control_test.cpp)
  target_link_libraries(wmfs-control-test PRIVATE wmfs::control)
  set_target_properties(wmfs-control-test PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
  add_test(NAME wmfs-control COMMAND wmfs-control-test)

  add_executable(wmfs-log-test tests/cpp/log_test.cpp)
  target_link_libraries(wmfs-log-test PRIVATE wmfs::log)
  set_target_properties(wmfs-log-test PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
  add_test(NAME wmfs-log COMMAND wmfs-log-test)

  add_executable(wmfs-unique-fd-test tests/cpp/unique_fd_test.cpp)
  target_include_directories(
    wmfs-unique-fd-test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  )
  add_test(NAME wmfs-unique-fd COMMAND wmfs-unique-fd-test)

  add_executable(wmfs-ring-protocol-test tests/cpp/ring_protocol_test.cpp)
  target_include_directories(
    wmfs-ring-protocol-test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  )
  set_target_properties(
    wmfs-ring-protocol-test
    PROPERTIES
      CXX_EXTENSIONS OFF
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON
  )
  add_test(NAME wmfs-ring-protocol COMMAND wmfs-ring-protocol-test)

  add_executable(wmfs-ring-test tests/cpp/ring_test.cpp)
  target_link_libraries(wmfs-ring-test PRIVATE wmfs::ring Threads::Threads)
  add_test(NAME wmfs-ring COMMAND wmfs-ring-test)
endif()

set(WMFS_BUNDLED_PLUGIN_NAMES "")
set(WMFS_BUNDLED_SOURCES "")
set(WMFS_BUNDLED_LIBRARIES "")
set(WMFS_BUNDLED_COMPILE_OPTIONS "")
set(WMFS_BUNDLED_INCLUDE_DIRECTORIES "")
set(WMFS_BUNDLED_PLUGIN_APIS "")
foreach(plugin IN LISTS WMFS_BUNDLED_PLUGINS)
  if(IS_ABSOLUTE "${plugin}")
    set(plugin_descriptor "${plugin}")
    get_filename_component(plugin_name "${plugin}" DIRECTORY)
    get_filename_component(plugin_name "${plugin_name}" NAME)
  else()
    set(plugin_descriptor "${CMAKE_CURRENT_SOURCE_DIR}/plugins/${plugin}/bundled.cmake")
    set(plugin_name "${plugin}")
  endif()
  if(NOT EXISTS "${plugin_descriptor}")
    message(FATAL_ERROR "Bundled plugin descriptor does not exist: ${plugin_descriptor}")
  endif()
  list(APPEND WMFS_BUNDLED_PLUGIN_NAMES "${plugin_name}")
  include("${plugin_descriptor}")
endforeach()
if(WMFS_BUNDLED_PLUGIN_NAMES AND NOT WMFS_BUILD_PYTHON_RUNTIME)
  message(FATAL_ERROR "Bundled plugins require the Python runtime")
endif()

if(WMFS_BUILD_REFERENCE_WORKER AND NOT TARGET wmfs-reference-kernels)
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
    PRIVATE wmfs::control
  )

  install(
    TARGETS wmfs-native-runtime
    LIBRARY DESTINATION wmfs
    COMPONENT runtime
  )

  if(WMFS_BUNDLED_PLUGIN_NAMES)
    set(WMFS_BUNDLED_REGISTRY_SOURCE
      "#include <cstring>\n#include <wmfs/bundled_registry.hpp>\n")
    foreach(plugin_api IN LISTS WMFS_BUNDLED_PLUGIN_APIS)
      string(REPLACE "|" ";" plugin_api_parts "${plugin_api}")
      list(GET plugin_api_parts 1 plugin_api_symbol)
      string(APPEND WMFS_BUNDLED_REGISTRY_SOURCE
        "extern \"C\" const wmfs_plugin_api_v1 *${plugin_api_symbol}(uint32_t);\n")
    endforeach()
    string(APPEND WMFS_BUNDLED_REGISTRY_SOURCE
      "const wmfs_plugin_api_v1 *wmfs_bundled_plugin_api(const char *name) {\n")
    foreach(plugin_api IN LISTS WMFS_BUNDLED_PLUGIN_APIS)
      string(REPLACE "|" ";" plugin_api_parts "${plugin_api}")
      list(GET plugin_api_parts 0 plugin_api_name)
      list(GET plugin_api_parts 1 plugin_api_symbol)
      string(APPEND WMFS_BUNDLED_REGISTRY_SOURCE
        "  if (std::strcmp(name, \"${plugin_api_name}\") == 0) return ${plugin_api_symbol}(WMFS_PLUGIN_ABI_VERSION);\n")
    endforeach()
    string(APPEND WMFS_BUNDLED_REGISTRY_SOURCE "  return 0;\n}\n")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bundled_registry.cpp"
      CONTENT "${WMFS_BUNDLED_REGISTRY_SOURCE}")
    list(JOIN WMFS_BUNDLED_PLUGIN_NAMES "," WMFS_BUNDLED_PLUGIN_NAMES_CSV)
    nanobind_add_module(
      wmfs-bundled-reference
      NB_STATIC
      src/bundled_module.cpp
      "${CMAKE_CURRENT_BINARY_DIR}/bundled_registry.cpp"
      ${WMFS_BUNDLED_SOURCES}
    )
    add_library(wmfs::bundled-reference ALIAS wmfs-bundled-reference)
    target_link_libraries(wmfs-bundled-reference PRIVATE ${WMFS_BUNDLED_LIBRARIES})
    target_include_directories(
      wmfs-bundled-reference
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
        ${WMFS_BUNDLED_INCLUDE_DIRECTORIES}
    )
    target_compile_options(wmfs-bundled-reference PRIVATE ${WMFS_BUNDLED_COMPILE_OPTIONS})
    target_compile_definitions(
      wmfs-bundled-reference
      PRIVATE WMFS_BUNDLED_PLUGIN_NAMES="${WMFS_BUNDLED_PLUGIN_NAMES_CSV}"
    )
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
  add_executable(
    wmfs-reference-worker
    src/reference_mapped_buffers.cpp
    src/reference_plugin_entry.cpp
    src/reference_worker.cpp
    plugins/reference/generated/src/reference_plugin_stub.cpp
  )
  target_include_directories(
    wmfs-reference-worker
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/generated"
      "${CMAKE_CURRENT_SOURCE_DIR}/plugins/reference/generated/include"
  )
  target_link_libraries(
    wmfs-reference-worker
    PRIVATE
      Threads::Threads
      wmfs::control
      wmfs::ring
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
      tests/cpp/reference_mapped_buffers_test.cpp
      src/reference_mapped_buffers.cpp
    )
    target_include_directories(
      wmfs-reference-mapped-buffers-test
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    )
    target_link_libraries(
      wmfs-reference-mapped-buffers-test
      PRIVATE wmfs::control Threads::Threads ${TORCH_LIBRARIES}
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
    DIRECTORY plugins/reference/generated/
    DESTINATION share/wmfs/plugins/reference/generated
    COMPONENT reference-worker
  )
endif()
