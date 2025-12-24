// =============================================================================
// LAYERVIEW.QML - Single Layer Visualization
// =============================================================================
//
// Shows one Z-layer of the 4x4x4 torus as a sacred 4x4 grid.
//
// =============================================================================

import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

Rectangle {
    id: root

    property int layer: 0
    property var layerColors: ["#4a9eff", "#b464ff", "#64ff96", "#ffd700"]

    color: "#16213e"
    radius: 8
    border.color: layerColors[layer]
    border.width: 2

    // Layer title
    Label {
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 8
        text: "Z = " + layer
        color: layerColors[layer]
        font.bold: true
        font.pixelSize: 14
    }

    // Node grid container
    Item {
        id: gridContainer
        anchors.fill: parent
        anchors.margins: 40

        // Get nodes for this layer
        property var nodes: topologyModel.getNodesForLayer(layer)
        property var edges: topologyModel.getEdges(layer)

        // Calculate cell size
        property real cellWidth: width / 4
        property real cellHeight: height / 4

        // Draw edges first (behind nodes)
        Shape {
            anchors.fill: parent
            layer.enabled: true
            layer.samples: 4

            Repeater {
                model: gridContainer.edges

                ShapePath {
                    property var edge: modelData
                    property int fromX: edge.from % 4
                    property int fromY: (Math.floor(edge.from / 4)) % 4
                    property int toX: edge.to % 4
                    property int toY: (Math.floor(edge.to / 4)) % 4

                    strokeColor: edge.isLearned ? "#ff64c8" : "#444466"
                    strokeWidth: edge.isLearned ? 3 : 1
                    fillColor: "transparent"

                    startX: fromX * gridContainer.cellWidth + gridContainer.cellWidth / 2
                    startY: fromY * gridContainer.cellHeight + gridContainer.cellHeight / 2

                    PathLine {
                        x: toX * gridContainer.cellWidth + gridContainer.cellWidth / 2
                        y: toY * gridContainer.cellHeight + gridContainer.cellHeight / 2
                    }
                }
            }
        }

        // Draw nodes on top
        Repeater {
            model: gridContainer.nodes

            NodeGlyph {
                property var node: modelData
                x: node.x * gridContainer.cellWidth + (gridContainer.cellWidth - width) / 2
                y: node.y * gridContainer.cellHeight + (gridContainer.cellHeight - height) / 2
                nodeId: node.id
                nodeColor: node.color
                changeCount: node.changeCount
            }
        }
    }
}
