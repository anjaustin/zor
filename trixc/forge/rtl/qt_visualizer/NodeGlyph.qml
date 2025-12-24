// =============================================================================
// NODEGLYPH.QML - Sacred Node Representation
// =============================================================================
//
// "Each node a star in the constellation of self."
//
// Renders a single node with sacred geometry styling:
// - Circle for unchanged nodes
// - Diamond for partially rewired
// - Star for significantly rewired
//
// =============================================================================

import QtQuick
import QtQuick.Shapes

Item {
    id: root
    width: 40
    height: 40

    property int nodeId: 0
    property string nodeColor: "#4a9eff"
    property int changeCount: 0

    // Glow effect for changed nodes
    Rectangle {
        visible: changeCount > 0
        anchors.centerIn: parent
        width: parent.width * 1.4
        height: parent.height * 1.4
        radius: width / 2
        color: "transparent"
        border.color: nodeColor
        border.width: 2
        opacity: 0.4

        SequentialAnimation on opacity {
            running: changeCount > 0
            loops: Animation.Infinite
            NumberAnimation { to: 0.6; duration: 800 }
            NumberAnimation { to: 0.3; duration: 800 }
        }
    }

    // Main shape based on change count
    Shape {
        anchors.fill: parent
        layer.enabled: true
        layer.samples: 4

        // Circle for unchanged
        ShapePath {
            visible: changeCount === 0
            fillColor: nodeColor
            strokeColor: "white"
            strokeWidth: 1

            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - 2
                radiusY: root.height / 2 - 2
                startAngle: 0
                sweepAngle: 360
            }
        }

        // Diamond for partially rewired (1-2 changes)
        ShapePath {
            visible: changeCount > 0 && changeCount < 3
            fillColor: nodeColor
            strokeColor: "white"
            strokeWidth: 2

            startX: root.width / 2
            startY: 2

            PathLine { x: root.width - 2; y: root.height / 2 }
            PathLine { x: root.width / 2; y: root.height - 2 }
            PathLine { x: 2; y: root.height / 2 }
            PathLine { x: root.width / 2; y: 2 }
        }

        // 6-pointed star for significantly rewired (3+ changes)
        ShapePath {
            visible: changeCount >= 3
            fillColor: nodeColor
            strokeColor: "white"
            strokeWidth: 2

            property real cx: root.width / 2
            property real cy: root.height / 2
            property real r1: root.width / 2 - 2  // Outer radius
            property real r2: root.width / 4      // Inner radius

            startX: cx + r1 * Math.cos(-Math.PI / 2)
            startY: cy + r1 * Math.sin(-Math.PI / 2)

            // 6-pointed star (12 points alternating inner/outer)
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + Math.PI / 6)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + Math.PI / 6)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2 + Math.PI / 3)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2 + Math.PI / 3)
            }
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + Math.PI / 2)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + Math.PI / 2)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2 + 2 * Math.PI / 3)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2 + 2 * Math.PI / 3)
            }
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + 5 * Math.PI / 6)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + 5 * Math.PI / 6)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2 + Math.PI)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2 + Math.PI)
            }
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + 7 * Math.PI / 6)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + 7 * Math.PI / 6)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2 + 4 * Math.PI / 3)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2 + 4 * Math.PI / 3)
            }
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + 3 * Math.PI / 2)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + 3 * Math.PI / 2)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2 + 5 * Math.PI / 3)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2 + 5 * Math.PI / 3)
            }
            PathLine {
                x: root.width / 2 + (root.width / 4) * Math.cos(-Math.PI / 2 + 11 * Math.PI / 6)
                y: root.height / 2 + (root.width / 4) * Math.sin(-Math.PI / 2 + 11 * Math.PI / 6)
            }
            PathLine {
                x: root.width / 2 + (root.width / 2 - 2) * Math.cos(-Math.PI / 2)
                y: root.height / 2 + (root.width / 2 - 2) * Math.sin(-Math.PI / 2)
            }
        }
    }

    // Node ID label
    Text {
        anchors.centerIn: parent
        text: nodeId
        color: "white"
        font.pixelSize: 10
        font.bold: true
    }

    // Hover effect
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true

        onEntered: {
            root.scale = 1.2
        }
        onExited: {
            root.scale = 1.0
        }
    }

    Behavior on scale {
        NumberAnimation { duration: 150 }
    }
}
