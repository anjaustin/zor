// =============================================================================
// METRICSPANEL.QML - Live Metrics Display
// =============================================================================
//
// "Watch the numbers. Feel the emergence."
//
// Displays:
// - Resonant count (with circular progress)
// - Global frustration (with bar)
// - Rewiring count
//
// =============================================================================

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Shapes

Rectangle {
    id: root
    color: "#16213e"
    radius: 8
    border.color: "#4a9eff"
    border.width: 1
    implicitHeight: 220

    Column {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 15

        Label {
            text: "METRICS"
            color: "#ffd700"
            font.bold: true
            font.pixelSize: 14
        }

        // Resonance circle
        Item {
            width: parent.width
            height: 80

            // Circular progress for resonance
            Shape {
                anchors.left: parent.left
                width: 70
                height: 70
                layer.enabled: true
                layer.samples: 4

                // Background circle
                ShapePath {
                    strokeColor: "#2a2a4e"
                    strokeWidth: 8
                    fillColor: "transparent"
                    capStyle: ShapePath.RoundCap

                    PathAngleArc {
                        centerX: 35
                        centerY: 35
                        radiusX: 27
                        radiusY: 27
                        startAngle: 0
                        sweepAngle: 360
                    }
                }

                // Progress arc
                ShapePath {
                    strokeColor: topologyModel.resonantCount === 64 ? "#64ff96" : "#ff64c8"
                    strokeWidth: 8
                    fillColor: "transparent"
                    capStyle: ShapePath.RoundCap

                    PathAngleArc {
                        centerX: 35
                        centerY: 35
                        radiusX: 27
                        radiusY: 27
                        startAngle: -90
                        sweepAngle: (topologyModel.resonantCount / 64.0) * 360
                    }
                }

                // Center text
                Text {
                    anchors.centerIn: parent
                    text: topologyModel.resonantCount
                    color: "white"
                    font.pixelSize: 18
                    font.bold: true
                }
            }

            Column {
                anchors.left: parent.left
                anchors.leftMargin: 85
                anchors.verticalCenter: parent.verticalCenter
                spacing: 4

                Label {
                    text: "RESONANCE"
                    color: "#a0a0ff"
                    font.pixelSize: 11
                }
                Label {
                    text: topologyModel.resonantCount + " / 64"
                    color: topologyModel.resonantCount === 64 ? "#64ff96" : "#ff64c8"
                    font.pixelSize: 16
                    font.bold: true
                }
                Label {
                    text: Math.round((topologyModel.resonantCount / 64.0) * 100) + "%"
                    color: "#888888"
                    font.pixelSize: 12
                }
            }
        }

        // Frustration bar
        Column {
            width: parent.width
            spacing: 4

            RowLayout {
                width: parent.width

                Label {
                    text: "FRUSTRATION"
                    color: "#a0a0ff"
                    font.pixelSize: 11
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: topologyModel.globalFrustration
                    color: topologyModel.globalFrustration === 0 ? "#64ff96" : "#ff6464"
                    font.bold: true
                }
            }

            Rectangle {
                width: parent.width
                height: 12
                radius: 6
                color: "#2a2a4e"

                Rectangle {
                    width: Math.min(parent.width, parent.width * (topologyModel.globalFrustration / 100.0))
                    height: parent.height
                    radius: 6
                    color: topologyModel.globalFrustration === 0 ? "#64ff96" :
                           topologyModel.globalFrustration < 20 ? "#ffd700" : "#ff6464"

                    Behavior on width {
                        NumberAnimation { duration: 200 }
                    }
                }
            }
        }

        // Rewiring count
        RowLayout {
            width: parent.width

            Label {
                text: "REWIRINGS"
                color: "#a0a0ff"
                font.pixelSize: 11
            }
            Item { Layout.fillWidth: true }
            Label {
                text: topologyModel.rewiringCount
                color: "#ff64c8"
                font.pixelSize: 16
                font.bold: true
            }
        }
    }
}
