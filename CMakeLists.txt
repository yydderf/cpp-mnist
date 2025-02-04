cmake_minimum_required(VERSION 3.18)
project(cpp_mnist LANGUAGES CXX)

if (NOT WIN32)
    string(ASCII 27 Esc)
    set(ColorReset  "${Esc}[m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DEPS_PREFIX ${CMAKE_BINARY_DIR}/_deps)
set(DEPS_INSTALL_PREFIX ${DEPS_PREFIX}/install)

list(APPEND CMAKE_PREFIX_PATH "${DEPS_INSTALL_PREFIX}")

include(ExternalProject)

find_package(BLAS QUIET)
find_package(Armadillo QUIET)

if (NOT OpenBLAS_FOUND)
    message(STATUS "${Yellow}OpenBLAS not found, fetch and build later${ColorReset}")
    ExternalProject_Add(
        OpenBLAS
        GIT_REPOSITORY  https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG         v0.3.29
        UPDATE_COMMAND  ""
        PREFIX          ${DEPS_PREFIX}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PREFIX}
            -DBUILD_SHARED_LIBS=ON
    )
    set(OpenBLAS_LIBRARIES "${DEPS_INSTALL_PREFIX}/lib/libopenblas.so")
    set(OpenBLAS_INCLUDE_DIRS "${DEPS_INSTALL_PREFIX}/include")
    set(OpenBLAS_FOUND TRUE)
else()
    message(STATUS "${Green}Found OpenBLAS: ${OpenBLAS_LIBRARIES}${ColorReset}")
endif()

function(build_armadillo)
    ExternalProject_Add(
        Armadillo
        GIT_REPOSITORY  https://gitlab.com/conradsnicta/armadillo-code.git
        GIT_TAG         14.2.x
        UPDATE_COMMAND  ""
        PREFIX          ${DEPS_PREFIX}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PREFIX}
        DEPENDS OpenBLAS
    )
    set(ARMADILLO_LIBRARIES "${DEPS_INSTALL_PREFIX}/lib/libarmadillo.so" PARENT_SCOPE)
    set(ARMADILLO_INCLUDE_DIRS "${DEPS_INSTALL_PREFIX}/include" PARENT_SCOPE)
    set(ARMADILLO_FOUND TRUE PARENT_SCOPE)
endfunction()

if (NOT Armadillo_FOUND)
    message(STATUS "${Yellow}Armadillo not found, fetch and build later${ColorReset}")
    build_armadillo()
else()
    message(STATUS "Found Armadillo: ${ARMADILLO_LIBRARIES}")
    execute_process(
        COMMAND ldd ${ARMADILLO_LIBRARIES}
        OUTPUT_VARIABLE ARMADILLO_LINKS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT ARMADILLO_LINKS MATCHES "libopenblas")
        message(STATUS "${Yellow}Armadillo does not link to OpenBLAS, rebuilding armadillo later${ColorReset}")
        build_armadillo()
    else()
        message(STATUS "${Green}Armadillo is correctly linked to OpenBLAS${ColorReset}")
    endif()
endif()

add_executable(main 
    src/main.cpp
    src/misc.cpp
)
add_dependencies(main
    Armadillo
)
target_include_directories(main
    PRIVATE ${ARMADILLO_INCLUDE_DIRS}
    PRIVATE ${OpenBLAS_INCLUDE_DIRS}
)
target_link_libraries(main
    ${ARMADILLO_LIBRARIES}
    ${OpenBLAS_LIBRARIES}
)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         2b6b042a77446ff322cd7522ca068d9f2a21c1d1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(misc_test
    tests/misc_test.cpp
    src/misc.cpp
)

target_link_libraries(misc_test
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(misc_test)
