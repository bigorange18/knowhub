cmake_minimum_required(VERSION 3.0)
project(main)


add_compile_options(--std=c++17)
# 需要找到Libtorch
# find_package(Torch REQUIRED)
find_package(GFlags REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Libtorch是基于C++14来实现的
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)