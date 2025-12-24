#include "ControlPanel.h"
#include "ZitEngine.h"
#include "FabricRenderer.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>

ControlPanel::ControlPanel(ZitEngine* engine, FabricRenderer* renderer,
                           QWidget* parent)
    : QWidget(parent)
    , m_engine(engine)
    , m_renderer(renderer)
{
    // BMW-inspired dark styling
    setStyleSheet(R"(
        QWidget {
            background-color: #14141a;
            color: white;
            font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
        }
        QPushButton {
            background-color: #1e1e28;
            border: 1px solid #333340;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #28283a;
            border-color: #1e90ff;
        }
        QPushButton:pressed {
            background-color: #1e90ff;
        }
        QPushButton#playButton {
            background-color: #1e90ff;
            border: none;
            font-weight: 600;
        }
        QPushButton#playButton:hover {
            background-color: #3ba0ff;
        }
        QSlider::groove:horizontal {
            background: #1e1e28;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #1e90ff;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #1e90ff;
            border-radius: 3px;
        }
        QCheckBox {
            font-size: 12px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid #333340;
            background: #1e1e28;
        }
        QCheckBox::indicator:checked {
            background: #1e90ff;
            border-color: #1e90ff;
        }
        QRadioButton {
            font-size: 12px;
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 1px solid #333340;
            background: #1e1e28;
        }
        QRadioButton::indicator:checked {
            background: #1e90ff;
            border: 3px solid #0a0a0f;
        }
        QLabel {
            font-size: 10px;
            color: #808080;
            text-transform: uppercase;
        }
        QGroupBox {
            border: 1px solid #1e1e28;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 8px;
        }
        QGroupBox::title {
            color: #808080;
            font-size: 10px;
        }
    )");

    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(16);
    layout->setContentsMargins(16, 16, 16, 16);

    // Playback controls
    auto* playLayout = new QHBoxLayout();
    playLayout->setSpacing(8);

    m_playButton = new QPushButton("Play");
    m_playButton->setObjectName("playButton");
    m_playButton->setFixedWidth(80);

    m_stepButton = new QPushButton("Step");
    m_stepButton->setFixedWidth(60);

    m_resetButton = new QPushButton("Reset");
    m_resetButton->setFixedWidth(60);

    playLayout->addWidget(m_playButton);
    playLayout->addWidget(m_stepButton);
    playLayout->addWidget(m_resetButton);
    playLayout->addStretch();

    // Speed slider
    auto* speedLabel = new QLabel("SPEED");
    m_speedSlider = new QSlider(Qt::Horizontal);
    m_speedSlider->setRange(1, 50);
    m_speedSlider->setValue(10);  // 1x speed

    // Color mode
    auto* colorLabel = new QLabel("COLOR BY");
    auto* colorLayout = new QHBoxLayout();
    colorLayout->setSpacing(16);

    m_colorModeGroup = new QButtonGroup(this);
    auto* rbResonance = new QRadioButton("Resonance");
    auto* rbResistance = new QRadioButton("Resistance");
    auto* rbState = new QRadioButton("State");

    rbResonance->setChecked(true);
    m_colorModeGroup->addButton(rbResonance, FabricRenderer::ByResonance);
    m_colorModeGroup->addButton(rbResistance, FabricRenderer::ByResistance);
    m_colorModeGroup->addButton(rbState, FabricRenderer::ByState);

    colorLayout->addWidget(rbResonance);
    colorLayout->addWidget(rbResistance);
    colorLayout->addWidget(rbState);
    colorLayout->addStretch();

    // Display options
    auto* displayLabel = new QLabel("DISPLAY");
    m_showEdgesCheck = new QCheckBox("Show Edges");
    m_showEdgesCheck->setChecked(true);

    // Add all to main layout
    layout->addLayout(playLayout);
    layout->addSpacing(8);
    layout->addWidget(speedLabel);
    layout->addWidget(m_speedSlider);
    layout->addSpacing(16);
    layout->addWidget(colorLabel);
    layout->addLayout(colorLayout);
    layout->addSpacing(16);
    layout->addWidget(displayLabel);
    layout->addWidget(m_showEdgesCheck);
    layout->addStretch();

    // Connections
    connect(m_playButton, &QPushButton::clicked, this, &ControlPanel::onPlayPause);
    connect(m_stepButton, &QPushButton::clicked, this, &ControlPanel::onStep);
    connect(m_resetButton, &QPushButton::clicked, this, &ControlPanel::onReset);
    connect(m_speedSlider, &QSlider::valueChanged, this, &ControlPanel::onSpeedChanged);
    connect(m_colorModeGroup, &QButtonGroup::idClicked, this, &ControlPanel::onColorModeChanged);
    connect(m_showEdgesCheck, &QCheckBox::toggled, this, &ControlPanel::onShowEdgesChanged);

    connect(m_engine, &ZitEngine::started, this, &ControlPanel::updatePlayButton);
    connect(m_engine, &ZitEngine::stopped, this, &ControlPanel::updatePlayButton);
    connect(m_engine, &ZitEngine::convergedSignal, this, &ControlPanel::updatePlayButton);
}

void ControlPanel::onPlayPause() {
    if (m_engine->isRunning()) {
        m_engine->stop();
    } else {
        m_engine->start();
    }
}

void ControlPanel::onStep() {
    if (!m_engine->isRunning() && !m_engine->converged()) {
        m_engine->step();
    }
}

void ControlPanel::onReset() {
    m_engine->reset();
    emit resetRequested();
}

void ControlPanel::onSpeedChanged(int value) {
    float speed = value / 10.0f;  // 0.1x to 5x
    m_engine->setSpeed(speed);
}

void ControlPanel::onColorModeChanged(int id) {
    m_renderer->setColorMode(static_cast<FabricRenderer::ColorMode>(id));
}

void ControlPanel::onShowEdgesChanged(bool checked) {
    m_renderer->setShowEdges(checked);
}

void ControlPanel::updatePlayButton() {
    if (m_engine->converged()) {
        m_playButton->setText("Done");
        m_playButton->setEnabled(false);
    } else if (m_engine->isRunning()) {
        m_playButton->setText("Pause");
        m_playButton->setEnabled(true);
    } else {
        m_playButton->setText("Play");
        m_playButton->setEnabled(true);
    }
}
