cmake_minimum_required(VERSION 3.16)
project(inkengine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(test_dequant tests/test_dequant.cpp)
target_include_directories(test_dequant PRIVATE include)
enable_testing()
add_test(NAME bit_exact_dequant
         COMMAND test_dequant ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors.bin)
