# =============================================================================
# SACRED TOPOLOGY VISUALIZER - Qt6 Build Configuration
# =============================================================================
#
# "The topology IS the self. Now we SEE the self."
#
# Build with:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x.x/gcc_64
#   make
#
# =============================================================================

cmake_minimum_required(VERSION 3.16)

project(SacredTopology VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find Qt6 packages
find_package(Qt6 REQUIRED COMPONENTS Quick QuickControls2 Widgets)

# Executable
qt_add_executable(sacred_topology
    main.cpp
    topologymodel.cpp
    topologymodel.h
)

# QML module
qt_add_qml_module(sacred_topology
    URI SacredTopology
    VERSION 1.0
    QML_FILES
        Main.qml
        TopologyView.qml
        NodeGlyph.qml
        LayerView.qml
        MetricsPanel.qml
    RESOURCES
        resources.qrc
)

# Link libraries
target_link_libraries(sacred_topology PRIVATE
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Widgets
)

# Install
install(TARGETS sacred_topology
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
