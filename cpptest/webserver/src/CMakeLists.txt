#指定cmake最低版本
cmake_minimum_required(VERSION 3.9)
#指定语言版本  这里指定C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#设置工程名字为main
project(main)
#设置工程名字为main, 版本为1.1
# project(main VERION 1.1)

# -------------------------------基本信息----------------------------------------
# 输出测试
message("hello")
# 打印项目名称
message(STATUS ${PROJECT_NAME})
# 打印项目路径
message(STATUS ${PROJECT_SOURCE_DIR})

# ------------------------------添加源文件------------------------------------------
#设定源码列表.cpp 假如你只有一两个源码文件
set(SOURCE_FILE ./main.cc)
#设定所有源码列表 
#方式一
#假如你有好多个源码文件放在./src目录下
#aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)  
#方式二
#设定所有源码列表 
# FILE（GLOB SRC_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp}） #匹配源码目录所有.cpp文件
# 递归查找src目录下所有cpp文件，包含src的子目录
FILE(GLOB_RECURSE SRC_LIST  ${PROJECT_SOURCE_DIR}/src/*.cpp)

#设定头文件路径
include_directories(../include/)
#inclde_directories("路径1"  “路径2”)



# 可执行文件
#add_executable(${PROJECT_NAME} ${SRC_LIST})  设置源码列表后，获得可执行文件这一步就很方便了，




#设定链接库的路径（一般使用第三方非系统目录下的库）  比如opencv??
link_directories(../build/)
link_directories("路径1"  “路径2)

#添加子目录，作用相当于进去子目录里面，展开子目录的CMakeLists.txt
#同时执行，子目录中的CMakeLists.txt一般是编译成一个库，作为一个模块
#在父目录中可以直接引用子目录生成的库
#add_subdirectory(math)

#生成动静态库
#add_library(动静态链接库名称  SHARED/STATIC(可选，默认STATIC)    源码列表)
add_libarary(model STATIC ${DIR}) # 生成libmodel.a
add_libarary(model SHARED ${DIR}) #生成libmodel.so

#生成可执行文件
add_executable(modle ${SOURCE_FILES})
target_link_libaries(${PROJECT_NAME} pthred glog) #就是g++ 编译选项中-l后的内容，不要有多余空格

ADD_CUSTOM_COMMAND(#执行shell命令
TARGET  modle
POST_BUILD # 在目标文件model build之后，执行下面的拷贝命令，还可以选择PRE_BUILD命令将会在其他依赖项执行前执行  PRE_LINK命令将会在其他依赖项执行完后执行   POST_BUILD命令将会在目标构建完执行
COMMAND cp  ./model ../
)