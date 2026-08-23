find_package(Torch CONFIG REQUIRED)

if(NOT TARGET wmfs-reference-kernels)
  add_library(wmfs-reference-kernels STATIC src/reference_kernels.cpp)
  set_target_properties(wmfs-reference-kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(wmfs-reference-kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  target_link_libraries(wmfs-reference-kernels PUBLIC ${TORCH_LIBRARIES})
  target_compile_options(wmfs-reference-kernels PRIVATE ${TORCH_CXX_FLAGS})
endif()

list(APPEND WMFS_BUNDLED_SOURCES
  src/reference_torch_ops.cpp
  src/reference_plugin_entry.cpp
  plugins/reference/generated/src/reference_plugin_stub.cpp
)
list(APPEND WMFS_BUNDLED_INCLUDE_DIRECTORIES plugins/reference/generated/include)
list(APPEND WMFS_BUNDLED_LIBRARIES wmfs-reference-kernels)
list(APPEND WMFS_BUNDLED_COMPILE_OPTIONS ${TORCH_CXX_FLAGS})
list(APPEND WMFS_BUNDLED_PLUGIN_APIS "reference|wmfs_reference_plugin_get_api")
