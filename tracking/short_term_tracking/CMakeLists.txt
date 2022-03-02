cmake_minimum_required(VERSION 3.0)
project (short_term_tracker)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)

set ( PROJECT_LINK_LIBS opencv_core.so opencv_highgui.so opencv_imgproc.so opencv_video.so opencv_videostab.so opencv_ml.so opencv_imgcodecs.so opencv_videoio.so opencv_calib3d cuda.so curand.so cudart.so)
link_directories( /usr/local/lib /usr/local/cuda/lib64 )

include_directories( /usr/local/include /usr/local/cuda/include )

add_executable(short_term_tracker short_term_tracker.cpp)
target_link_libraries(short_term_tracker ${PROJECT_LINK_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET short_term_tracker PROPERTY CXX_STANDARD 14)



