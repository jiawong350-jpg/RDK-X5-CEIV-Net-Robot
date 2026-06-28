cmake_minimum_required(VERSION 3.10)
project(rdk_ai_module)

set(CMAKE_CXX_STANDARD 14)

# 1. 查找 OpenCV (图像处理)
find_package(OpenCV REQUIRED)

# 2. 查找 Pybind11 (Python 绑定)
# 如果你的板子上没有安装 pybind11，需要先 pip install pybind11
find_package(pybind11 REQUIRED)

# 3. 包含头文件目录
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 4. 🔥 核心：生成 Python 扩展模块 (.so) 🔥
# 注意：这里千万不要加 main.cpp！
# 只需要我们刚才修改的那三个核心文件
pybind11_add_module(rdk_ai_module
    src/bridge.cpp
    src/face_engine.cpp
    src/hand_engine.cpp
)

# 5. 链接库
target_link_libraries(rdk_ai_module PRIVATE ${OpenCV_LIBS})

# 优化编译选项 (可选，让代码跑得更快)
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(rdk_ai_module PRIVATE -O3 -pthread)
endif()