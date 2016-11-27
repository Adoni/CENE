# Install script for directory: /Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/cnn" TYPE FILE FILES
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/aligned-mem-pool.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/cfsm-builder.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/c2w.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/cnn.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/conv.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/cuda.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/dict.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/dim.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/exec.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/expr.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/fast-lstm.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/functors.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/gpu-kernels.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/gpu-ops.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/graph.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/gru.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/init.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/lstm.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/model.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/mp.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/nodes.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/param-nodes.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/random.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/rnn-state-machine.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/rnn.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/saxe-init.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/shadow-params.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/simd-functors.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/tensor.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/timing.h"
    "/Users/sunxiaofei/ClionProjects/DLNE/cnn/cnn/training.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/sunxiaofei/ClionProjects/DLNE/cmake-build-debug/cnn/cnn/libcnn.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcnn.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcnn.a")
    execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcnn.a")
  endif()
endif()

