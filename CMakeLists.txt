cmake_minimum_required(VERSION 3.28)
project(mini_serve LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# kv-cache-bench's static libraries end up inside our shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The attention kernel: chipontix/kv-cache-bench pinned to commit 1be44cd. Only its kvmetal
# library (paged decode attention on Metal) is built; its tests and benchmark are not.
include(FetchContent)
FetchContent_Declare(kvbench
  URL https://github.com/chipontix/kv-cache-bench/archive/1be44cdeeb86debc5ff80e417f114f59fb2d1500.tar.gz
  URL_HASH SHA256=2bbfddf0de99cf6608da91484d43be717fcb8fd24a18bd6f1edfd4331629adc5
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(kvbench)

# The model runner, loaded from Python with ctypes (mini_serve/engine.py).
add_library(miniserve SHARED src/engine.cpp)
target_compile_options(miniserve PRIVATE -Wall -Wextra)
target_compile_definitions(miniserve PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(miniserve PRIVATE kvmetal "-framework Accelerate")
