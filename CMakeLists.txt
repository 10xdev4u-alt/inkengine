cmake_minimum_required(VERSION 3.16)
project(inkengine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)

configure_file(tests/vectors.bin ${CMAKE_CURRENT_BINARY_DIR}/vectors.bin COPYONLY)
configure_file(tests/vectors_mxfp8.bin ${CMAKE_CURRENT_BINARY_DIR}/vectors_mxfp8.bin COPYONLY)
configure_file(tests/vectors_mxfp8_2d.bin ${CMAKE_CURRENT_BINARY_DIR}/vectors_mxfp8_2d.bin COPYONLY)
foreach(fixture IN ITEMS fixture_header.json fixture_codes.bin fixture_scales.bin)
  configure_file(tests/${fixture} ${CMAKE_CURRENT_BINARY_DIR}/${fixture} COPYONLY)
endforeach()

add_executable(test_dequant tests/test_dequant.cpp)
target_include_directories(test_dequant PRIVATE include)

add_executable(test_codec tests/test_codec.cpp)
target_include_directories(test_codec PRIVATE include)
target_link_libraries(test_codec PRIVATE ZLIB::ZLIB)

enable_testing()
add_test(NAME bit_exact_dequant
         COMMAND test_dequant ${CMAKE_CURRENT_BINARY_DIR}/vectors.bin
                              ${CMAKE_CURRENT_BINARY_DIR}/vectors_mxfp8.bin
                              ${CMAKE_CURRENT_BINARY_DIR}/vectors_mxfp8_2d.bin)
add_test(NAME codec_reader
         COMMAND test_codec ${CMAKE_CURRENT_BINARY_DIR}/fixture_header.json
                            ${CMAKE_CURRENT_BINARY_DIR}/fixture_codes.bin
                            ${CMAKE_CURRENT_BINARY_DIR}/fixture_scales.bin)
