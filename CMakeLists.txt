cmake_minimum_required(VERSION 3.18)
project(cpp_mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         2b6b042a77446ff322cd7522ca068d9f2a21c1d1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(main 
    src/main.cpp
)
target_link_libraries(main ${ARMADILLO_LIBRARIES})

enable_testing()

add_executable(
    misc_test
    tests/misc_test.cpp
)

target_link_libraries(
    misc_test
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(misc_test)
