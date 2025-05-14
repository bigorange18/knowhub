### CMakeLists

### 常用宏

例如**Server**项目的工程目录：

![image-20250514082335694](../.assert/image-20250514082335694.png)

​		

|         宏         |                      说明                      |
| :----------------: | :--------------------------------------------: |
|    PROJECT_NAME    |                   项目工程名                   |
| PROJECT_SOURCE_DIR | 工程源码目录，就是CMakeLists.txt所在Server目录 |
|  CAMKE_SOURCE_DIR  |             等价PROJECT_SOURCE_DIR             |
|    _SOURCE_DIR     |             等价PROJECT_SOURCE_DIR             |
| PROJECT_BINARY_DIR |        生成文件目录，例如~/Server/build        |
|  CAMKE_BINARY_DIR  |             等价PROJECT_BINARY_DIR             |
|    _BINARY_DIR     |                                                |
|                    |                                                |
|                    |                                                |
|                    |                                                |
|                    |                                                |
|                    |                                                |

```cmake
cmake_minimum_required(VERSION 3.10)
# ------------------------------基本信息------------------------------------------
set(CMAKE_CXX_STANDARD 17)	# C++17
# add_compile_options(--std=c++17)
project(main  VERSION 1.1 LANGUAGES CXX)
message(STATUS "Start build...")
# message(${PROJECT_NAME})
message(STATUS ${PROJECT_BINARY_DIR})
# message(STATUS ${PROJECT_VERSION})
message(STATUS ${CMAKE_SOURCE_DIR}/${PROJECT_VERSION})
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
# message(${source})
# set(SRC main.cc src/server.cc src/server.h)
# ------------------------------添加源文件------------------------------------------
# 方式一:查找源文件
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)
aux_source_directory(${PROJECT_SOURCE_DIR}/include HEADER_LIST)
# # 方式二:查找源文件
# file(GLOB CC_SRC ${PROJECT_SOURCE_DIR}/src/*.cc)
# file(GLOB HEAD_SRC ${PROJECT_SOURCE_DIR}/src/*.h)
include_directories(${PROJECT_SOURCE_DIR})

# ------------------------------链接第三方库------------------------------------------

# 添加PCL环境
find_package(OpenCV REQUIRED)
# 打印opencv信息
# message(STATUS "OpenCV library status:")
# message(STATUS "    config: ${OpenCV_DIR}")
# message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${OpenCV_LIBS}")
# message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# ------------------------------可执行文件------------------------------------------
add_executable(${PROJECT_NAME}  main.cc ${SRC_LIST} ${HEADER_LIST})
# 补充链接库
target_link_libraries(${PROJECT_NAME} PRIVATE 
	${OPENCV_LIBS} 
)


#将编译文件指定到输出的路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${PROJECT_VERSION})
# 添加头文件的目录, 如果cc文件中引用的了相对路径的头文件,那么在包含头文件时候，要添加相对路径的前半部分

# 将config.h.in 文件里面@xx@ 中的xx替换为版本号
configure_file(${PROJECT_SOURCE_DIR}/config/config.h.in ${PROJECT_SOURCE_DIR}/config/config.h)

option(DATE_FLAG "update date" ON)
if(DATE_FLAG)
    set(DATE "2025.05.10")
endif()


# 制作静态库文件
# add_library(calc STATIC ${SRC_LIST})
# 制作动态库文件
# add_library(calc SHARED ${SRC_LIST})

```

