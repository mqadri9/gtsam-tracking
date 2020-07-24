project(test)
cmake_minimum_required(VERSION 3.0)

set(EXTERNAL_LIBS "")
set(EXTRA_DEPS "")

find_package( libpointmatcher REQUIRED )
include_directories(${libpointmatcher_INCLUDE_DIRS})
set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${libpointmatcher_LIBRARIES})

# Suppress Eigen's warning by adding it to the system's library
#find_package(Eigen3 REQUIRED)
#include_directories(SYSTEM "${EIGEN_INCLUDE_DIR}")
#set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${Eigen3_LIBRARIES})

#find_package(libnabo REQUIRED PATHS ${LIBNABO_INSTALL_DIR})
#include(libnaboConfig)
#include_directories(${libnabo_INCLUDE_DIRS})
#set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${libnabo_LIBRARIES})
#message(STATUS "libnabo found, version ${libnabo_VERSION} (include=${libnabo_INCLUDE_DIRS} libs=${libnabo_LIBRARIES})")

include_directories(..)

find_package(Boost COMPONENTS thread filesystem system program_options date_time REQUIRED)
if (Boost_MINOR_VERSION GREATER 47)
    find_package(Boost COMPONENTS thread filesystem system program_options date_time chrono REQUIRED)
endif (Boost_MINOR_VERSION GREATER 47)
include_directories(${Boost_INCLUDE_DIRS})
set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${Boost_LIBRARIES})

#message("${EXTERNAL_LIBS}")

# Find GTSAM components
find_package(GTSAMCMakeTools)
find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})

set(EXTERNAL_LIBS ${EXTERNAL_LIBS} gtsam)

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIR})
set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${OpenCV_LIBS})

message("${EXTERNAL_LIBS}")

add_executable(main main.cpp Procrustes.cpp utils.cpp pointcloud.cpp optimizer.cpp test_sfm.cpp pose.cpp
               ./Procrustes.h ./utils.h ./pointcloud.h ./optimizer.h ./test_sfm.h ./pose.h)
add_executable(procrustes test_procrustes.cpp Procrustes.cpp ./Procrustes.h)

target_link_libraries(main ${EXTERNAL_LIBS})
target_link_libraries(procrustes ${EXTERNAL_LIBS})





