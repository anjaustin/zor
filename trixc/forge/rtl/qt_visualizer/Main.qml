// =============================================================================
// MAIN.QML - Sacred Topology Visualizer Main Window
// =============================================================================
//
// "See the self. Watch it learn. Witness emergence."
//
// =============================================================================

import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1400
    height: 900
    title: "Sacred Topology Visualizer - Hollywood Squares"

    Material.theme: Material.Dark
    Material.accent: Material.Amber
    Material.primary: Material.Indigo

    color: "#1a1a2e"

    // Header
    header: ToolBar {
        Material.background: "#16213e"
        height: 60

        RowLayout {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 20

            Label {
                text: "THE TOPOLOGY IS THE SELF"
                font.pixelSize: 24
                font.bold: true
                color: "#ffd700"
            }

            Item { Layout.fillWidth: true }

            // Cycle counter
            Label {
                text: "Cycle: " + topologyModel.currentCycle + " / " + (topologyModel.totalCycles - 1)
                font.pixelSize: 16
                color: "#a0a0ff"
            }
        }
    }

    // Main content
    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        // Left side: 4 layer views in 2x2 grid
        GridLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width * 0.7
            columns: 2
            rowSpacing: 10
            columnSpacing: 10

            Repeater {
                model: 4
                delegate: LayerView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    layer: index
                    layerColors: ["#4a9eff", "#b464ff", "#64ff96", "#ffd700"]
                }
            }
        }

        // Right side: Metrics and controls
        ColumnLayout {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            spacing: 20

            // Metrics Panel
            MetricsPanel {
                Layout.fillWidth: true
            }

            // 3D Overview (simplified)
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                color: "#16213e"
                radius: 8
                border.color: "#4a9eff"
                border.width: 1

                Column {
                    anchors.centerIn: parent
                    spacing: 10

                    Label {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: "3D TORUS"
                        color: "#ffd700"
                        font.bold: true
                    }

                    // Simple 3D representation
                    TopologyView {
                        width: 180
                        height: 140
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }
            }

            // Playback controls
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                color: "#16213e"
                radius: 8
                border.color: "#4a9eff"
                border.width: 1

                Column {
                    anchors.fill: parent
                    anchors.margins: 15
                    spacing: 10

                    Label {
                        text: "PLAYBACK"
                        color: "#ffd700"
                        font.bold: true
                    }

                    // Timeline slider
                    Slider {
                        width: parent.width
                        from: 0
                        to: Math.max(0, topologyModel.totalCycles - 1)
                        value: topologyModel.currentCycle
                        stepSize: 1
                        onMoved: topologyModel.currentCycle = value

                        Material.accent: "#ff64c8"
                    }

                    // Control buttons
                    RowLayout {
                        width: parent.width
                        spacing: 5

                        Button {
                            text: "|<"
                            flat: true
                            onClicked: topologyModel.goToStart()
                        }
                        Button {
                            text: "<"
                            flat: true
                            onClicked: topologyModel.stepBackward()
                        }
                        Button {
                            text: topologyModel.isPlaying ? "||" : ">"
                            flat: true
                            font.pixelSize: 18
                            onClicked: topologyModel.isPlaying ? topologyModel.pause() : topologyModel.play()
                        }
                        Button {
                            text: ">"
                            flat: true
                            onClicked: topologyModel.stepForward()
                        }
                        Button {
                            text: ">|"
                            flat: true
                            onClicked: topologyModel.goToEnd()
                        }
                    }
                }
            }

            // Legend
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 180
                color: "#16213e"
                radius: 8
                border.color: "#4a9eff"
                border.width: 1

                Column {
                    anchors.fill: parent
                    anchors.margins: 15
                    spacing: 8

                    Label {
                        text: "LEGEND"
                        color: "#ffd700"
                        font.bold: true
                    }

                    // Layer colors
                    Repeater {
                        model: [
                            { color: "#4a9eff", label: "Layer Z=0 (Unchanged)" },
                            { color: "#b464ff", label: "Layer Z=1 (Unchanged)" },
                            { color: "#64ff96", label: "Layer Z=2 (Unchanged)" },
                            { color: "#ffd700", label: "Layer Z=3 (Unchanged)" },
                            { color: "#ff64c8", label: "Partially Rewired" },
                            { color: "#ffd700", label: "Significantly Rewired" }
                        ]
                        delegate: Row {
                            spacing: 10
                            Rectangle {
                                width: 16
                                height: 16
                                radius: 8
                                color: modelData.color
                                border.color: "white"
                                border.width: 1
                            }
                            Label {
                                text: modelData.label
                                color: "#cccccc"
                                font.pixelSize: 11
                            }
                        }
                    }
                }
            }

            Item { Layout.fillHeight: true }
        }
    }

    // Keyboard shortcuts
    Shortcut {
        sequence: "Space"
        onActivated: topologyModel.isPlaying ? topologyModel.pause() : topologyModel.play()
    }
    Shortcut {
        sequence: "Left"
        onActivated: topologyModel.stepBackward()
    }
    Shortcut {
        sequence: "Right"
        onActivated: topologyModel.stepForward()
    }
    Shortcut {
        sequence: "Home"
        onActivated: topologyModel.goToStart()
    }
    Shortcut {
        sequence: "End"
        onActivated: topologyModel.goToEnd()
    }
}
