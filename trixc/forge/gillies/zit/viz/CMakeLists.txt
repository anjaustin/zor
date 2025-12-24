cmake_minimum_required(VERSION 3.16)
project(ZitViz VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL OpenGLWidgets)

# ZIT library (the C implementation)
add_library(zit STATIC
    ../zit.c
    ../zit.h
)
target_include_directories(zit PUBLIC ..)

# Visualization application
add_executable(zit_viz
    main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/ZitEngine.cpp
    src/ZitEngine.h
    src/FabricRenderer.cpp
    src/FabricRenderer.h
    src/MetricsPanel.cpp
    src/MetricsPanel.h
    src/ControlPanel.cpp
    src/ControlPanel.h
    resources/resources.qrc
)

target_link_libraries(zit_viz PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    zit
    m
)

target_include_directories(zit_viz PRIVATE src)

# Copy shaders to build directory
file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})

# Install target
install(TARGETS zit_viz DESTINATION bin)
