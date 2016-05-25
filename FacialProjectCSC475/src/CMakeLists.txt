cmake_minimum_required(VERSION 2.8)
project( facialproject )
find_package( OpenCV REQUIRED )
set (facialproject_VERSION_MAJOR 1)
# configure a header file to pass some of the CMake settings
# to the source code
#configure_file (
#  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
#  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
#  )
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( facialproject main.cpp fn.h)
target_link_libraries( facialproject ${OpenCV_LIBS} )
