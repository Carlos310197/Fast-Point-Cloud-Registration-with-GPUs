cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(icp LANGUAGES CXX CUDA)

include(CTest)

include_directories("/usr/local/cuda-11.6/samples/Common") # CUDA 11.6
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_library(icp_lib STATIC
            src/my_lib.cpp
            src/my_lib.h
            )

# Request that icp_lib be built with -std=c++11
# As this is a public compile feature anything that links to icp_lib
# will also build with -std=c++11
target_compile_features(icp_lib PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(icp_lib
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)

  add_executable(icp_test src/ICP_standard.cu)

  set_target_properties(icp_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(icp_test PRIVATE icp_lib -lcublas -lcurand -lcusolver)

  add_test(NAME icp_test COMMAND icp_test)

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET icp_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()

endif()