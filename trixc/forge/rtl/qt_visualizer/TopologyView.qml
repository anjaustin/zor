// =============================================================================
// TOPOLOGYVIEW.QML - 3D Overview (Isometric Projection)
// =============================================================================
//
// "See the torus. See the self in space."
//
// Shows a simplified 3D isometric view of the 4x4x4 topology.
// Layers stacked with perspective.
//
// =============================================================================

import QtQuick
import QtQuick.Shapes

Item {
    id: root

    // Isometric projection parameters
    property real layerOffset: 15  // Vertical offset between layers
    property real scale: 0.8
    property var layerColors: ["#4a9eff", "#b464ff", "#64ff96", "#ffd700"]

    // Draw layers from back to front
    Repeater {
        model: 4

        Item {
            property int layerZ: 3 - index  // Draw back to front
            y: index * root.layerOffset

            // Layer nodes
            Repeater {
                model: topologyModel.getNodesForLayer(layerZ)

                Rectangle {
                    property var node: modelData
                    property real cellSize: (root.width - 40) / 4

                    x: 20 + node.x * cellSize
                    y: node.y * cellSize * 0.6  // Compress Y for isometric
                    width: 8
                    height: 8
                    radius: 4

                    color: node.changeCount === 0 ? layerColors[layerZ] :
                           node.changeCount < 3 ? "#ff64c8" : "#ffd700"

                    border.color: "white"
                    border.width: node.changeCount > 0 ? 1 : 0

                    opacity: 0.7 + (layerZ * 0.1)
                }
            }

            // Layer outline
            Rectangle {
                x: 15
                y: -5
                width: root.width - 30
                height: (root.width - 30) * 0.6
                color: "transparent"
                border.color: layerColors[layerZ]
                border.width: 1
                opacity: 0.3
                radius: 4
            }
        }
    }

    // Animation for a subtle rotation effect
    transform: Rotation {
        id: rotationTransform
        origin.x: root.width / 2
        origin.y: root.height / 2
        axis { x: 0.3; y: 1; z: 0 }
        angle: 0

        SequentialAnimation on angle {
            running: true
            loops: Animation.Infinite
            NumberAnimation { to: 10; duration: 3000; easing.type: Easing.InOutSine }
            NumberAnimation { to: -10; duration: 3000; easing.type: Easing.InOutSine }
        }
    }
}
