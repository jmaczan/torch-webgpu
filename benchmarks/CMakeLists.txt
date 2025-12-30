cmake_minimum_required(VERSION 3.22)
project(torch_webgpu_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dawn path: inherit from env or pass -DDAWN_PREFIX=/path/to/dawn/install
if(NOT DEFINED DAWN_PREFIX)
  if(DEFINED ENV{DAWN_PREFIX})
    set(DAWN_PREFIX $ENV{DAWN_PREFIX})
  else()
    message(FATAL_ERROR "DAWN_PREFIX not set; export DAWN_PREFIX or pass -DDAWN_PREFIX=/path")
  endif()
endif()

# Torch C++ SDK path
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebench
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googlebench)

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

add_executable(bench_webgpu bench_mm.cpp)
target_link_libraries(bench_webgpu PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,torch_webgpu_core> benchmark::benchmark Python3::Python)
