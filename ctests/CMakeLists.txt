cmake_minimum_required(VERSION 3.22)
project(torch_webgpu_ctests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development)

include(FetchContent)
set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

file(GLOB_RECURSE TORCH_WEBGPU_SRC CONFIGURE_DEPENDS ../csrc/*.cpp)

add_library(torch_webgpu_core ${TORCH_WEBGPU_SRC})
target_include_directories(torch_webgpu_core PUBLIC
  ${PROJECT_SOURCE_DIR}/../csrc
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${DAWN_PREFIX}/include
)
target_link_directories(torch_webgpu_core PUBLIC ${DAWN_PREFIX}/lib)
target_link_libraries(torch_webgpu_core PUBLIC ${TORCH_LIBRARIES} webgpu_dawn)
target_compile_options(torch_webgpu_core PRIVATE ${TORCH_CXX_FLAGS})

enable_testing()

file(GLOB ctest_sources CONFIGURE_DEPENDS *.cpp)
foreach(test_src ${ctest_sources})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,torch_webgpu_core> GTest::gtest_main Python3::Python)
  gtest_discover_tests(${test_name})
endforeach()
