cmake_minimum_required(VERSION 3.0)
project (mrc)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -lpthread")

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

set ( PROJECT_LINK_LIBS opencv_core.so opencv_highgui.so opencv_imgproc.so opencv_video.so opencv_videostab.so opencv_ml.so opencv_imgcodecs.so opencv_videoio.so lcm.so X11.so  cuda.so curand.so cudart.so pthread.so)

link_directories( /usr/local/lib /usr/local/cuda/lib64 )

include_directories( /usr/local/include /usr/local/cuda/include /usr/local/cuda/targets/x86_64-linux/include )

add_executable(mrc mrc.cpp)
target_link_libraries(mrc ${PROJECT_LINK_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET mrc PROPERTY CXX_STANDARD 14)



