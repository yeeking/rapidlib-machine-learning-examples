cmake_minimum_required (VERSION 3.10)

# Define the project
project(rapidlib-regession-example C CXX)

# this adds all h and cpp files from lib folder to the 
# build
# apparently you're not supposed to do this after Cmake 3.12 
# but it works 
file(GLOB LIB_DIR
    "../lib/*.h"
    "../lib/*.cpp"
)

# this adds the pthreads library (g++ -pthread)
FIND_PACKAGE ( Threads REQUIRED )

# build the rapidlib library code
add_library(rapidlib ${LIB_DIR})

# build the executable
add_executable(rapidlib-regression-example src/main.cpp)
# link the main exutable to the rapidlib library and pthreads
target_link_libraries(rapidlib-regression-example rapidlib ${CMAKE_THREAD_LIBS_INIT})
