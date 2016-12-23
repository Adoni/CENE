cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(DLNE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -std=c++11")

set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if(NOT APPLE)
    find_package(Threads REQUIRED)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()
#include_directories(${include_directories} cnn)

#add_subdirectory(cnn/cnn)

get_filename_component(DYNET_INCLUDE_DIR "${DYNET_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
include_directories(${DYNET_INCLUDE_DIR})
message("-- Dynet dir is " ${DYNET_INCLUDE_DIR})

get_filename_component(DYNET_LINK_DIR "${DYNET_LINK_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
link_directories(${DYNET_LINK_DIR})
message("-- Dynet link is " ${DYNET_LINK_DIR})

foreach (TARGET function_test)
    ADD_EXECUTABLE(${TARGET} src/${TARGET}.cpp)
    target_link_libraries(${TARGET} dynet ${LIBS})
endforeach ()

set(DLNE_MAIN_SOURCE
        src/dlne_main.cpp
        src/mp_train.cpp
        src/mp_train.h
        src/sp_train.h
        src/embedding_methods.h
        src/network_embedding.h
        src/network_data.h
        )
add_executable(dlne_main ${DLNE_MAIN_SOURCE})
target_link_libraries(dlne_main dynet ${LIBS})
if (UNIX AND NOT APPLE)
    target_link_libraries(dlne_main rt)
endif ()
