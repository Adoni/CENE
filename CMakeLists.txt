cmake_minimum_required(VERSION 2.8)
project(DLNE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -std=c++11")

if (DEFINED ENV{BOOST_ROOT})
    set(Boost_NO_SYSTEM_PATHS ON)
endif ()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#find_package(Threads REQUIRED)
find_package(Threads)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

include_directories(${include_directories} cnn)

add_subdirectory(cnn/cnn)

foreach (TARGET function_test)
    ADD_EXECUTABLE(${TARGET} src/${TARGET}.cpp)
    target_link_libraries(${TARGET} cnn ${LIBS} pthread)
endforeach ()

set(DLNE_MAIN_SOURCE
        src/dlne_main.cpp
        src/mp_train
        src/sp_train
        )
add_executable(dlne_main ${DLNE_MAIN_SOURCE})
target_link_libraries(dlne_main cnn ${LIBS} pthread)
if (UNIX AND NOT APPLE)
    target_link_libraries(dlne_main rt)
endif ()