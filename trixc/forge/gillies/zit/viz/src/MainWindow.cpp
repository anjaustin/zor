#include "MainWindow.h"
#include "ZitEngine.h"
#include "FabricRenderer.h"
#include "MetricsPanel.h"
#include "ControlPanel.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSplitter>
#include <QKeyEvent>
#include <QLabel>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("ZIT Fabric Visualizer");
    setMinimumSize(1200, 800);

    // Dark theme for main window
    setStyleSheet(R"(
        QMainWindow {
            background-color: #0a0a0f;
        }
        QSplitter::handle {
            background-color: #1e1e28;
            width: 1px;
        }
    )");

    // Create engine (8x8x8 = 512 nodes)
    m_engine = new ZitEngine(8, this);

    // Central widget with splitter
    auto* central = new QWidget(this);
    auto* mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // Left side: Controls panel
    auto* leftPanel = new QWidget();
    leftPanel->setFixedWidth(280);
    leftPanel->setStyleSheet("background-color: #14141a;");
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(0);

    // Title bar
    auto* titleBar = new QWidget();
    titleBar->setFixedHeight(60);
    titleBar->setStyleSheet("background-color: #0a0a0f;");
    auto* titleLayout = new QVBoxLayout(titleBar);
    auto* titleLabel = new QLabel("ZIT FABRIC");
    titleLabel->setStyleSheet(R"(
        font-size: 16px;
        font-weight: 600;
        color: white;
        font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
        letter-spacing: 2px;
    )");
    auto* subtitleLabel = new QLabel("Homeo-Adaptive Topology");
    subtitleLabel->setStyleSheet(R"(
        font-size: 10px;
        color: #808080;
        font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
    )");
    titleLayout->addWidget(titleLabel);
    titleLayout->addWidget(subtitleLabel);

    // Need to create renderer first for control panel
    m_renderer = new FabricRenderer(m_engine, this);

    m_controls = new ControlPanel(m_engine, m_renderer, this);
    m_metrics = new MetricsPanel(m_engine, this);

    // Connect reset signal
    connect(m_controls, &ControlPanel::resetRequested, m_metrics, &MetricsPanel::reset);
    connect(m_controls, &ControlPanel::resetRequested, m_renderer, &FabricRenderer::resetCamera);

    leftLayout->addWidget(titleBar);
    leftLayout->addWidget(m_controls);
    leftLayout->addWidget(m_metrics, 1);

    // Center: 3D view
    m_renderer->setMinimumWidth(600);

    // Splitter
    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(leftPanel);
    splitter->addWidget(m_renderer);
    splitter->setStretchFactor(0, 0);  // Left panel doesn't stretch
    splitter->setStretchFactor(1, 1);  // 3D view stretches

    mainLayout->addWidget(splitter);

    setCentralWidget(central);

    // Initial state update
    m_metrics->updateMetrics();
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Space:
            if (m_engine->isRunning()) {
                m_engine->stop();
            } else {
                m_engine->start();
            }
            break;

        case Qt::Key_R:
            m_engine->reset();
            m_metrics->reset();
            m_renderer->resetCamera();
            break;

        case Qt::Key_1:
            m_renderer->setColorMode(FabricRenderer::ByState);
            break;

        case Qt::Key_2:
            m_renderer->setColorMode(FabricRenderer::ByResistance);
            break;

        case Qt::Key_3:
            m_renderer->setColorMode(FabricRenderer::ByResonance);
            break;

        case Qt::Key_E:
            m_renderer->setShowEdges(!m_renderer->showEdges());
            break;

        default:
            QMainWindow::keyPressEvent(event);
    }
}
