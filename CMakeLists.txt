cmake_minimum_required(VERSION 3.18)
project(cpp_mnist LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(main 
    src/main.cpp
    src/misc.cpp)
target_link_libraries(main ${ARMADILLO_LIBRARIES})
