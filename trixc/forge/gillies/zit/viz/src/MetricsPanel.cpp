#include "MetricsPanel.h"
#include "ZitEngine.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>

// BMW-inspired colors
static const QColor COLOR_BG(20, 20, 26);
static const QColor COLOR_SURFACE(30, 30, 40);
static const QColor COLOR_ACCENT(30, 144, 255);
static const QColor COLOR_TEXT(255, 255, 255);
static const QColor COLOR_TEXT_DIM(128, 128, 128);
static const QColor COLOR_SUCCESS(0, 212, 170);
static const QColor COLOR_CHART_LINE(30, 144, 255);
static const QColor COLOR_CHART_FILL(30, 144, 255, 40);

// ========== ResonanceChart ==========

ResonanceChart::ResonanceChart(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(80);
    setMaximumHeight(100);
}

void ResonanceChart::addDataPoint(int cycle, float resonanceRatio) {
    m_data.append(QPointF(cycle, resonanceRatio));
    if (m_data.size() > MAX_POINTS) {
        m_data.removeFirst();
    }
    update();
}

void ResonanceChart::reset() {
    m_data.clear();
    update();
}

void ResonanceChart::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // Background
    p.fillRect(rect(), COLOR_SURFACE);

    if (m_data.size() < 2) return;

    // Calculate bounds
    float minX = m_data.first().x();
    float maxX = m_data.last().x();
    if (maxX == minX) maxX = minX + 1;

    float w = width() - 4;
    float h = height() - 4;

    // Build path
    QPainterPath path;
    bool first = true;
    for (const QPointF& pt : m_data) {
        float x = 2 + (pt.x() - minX) / (maxX - minX) * w;
        float y = 2 + (1.0f - pt.y()) * h;
        if (first) {
            path.moveTo(x, y);
            first = false;
        } else {
            path.lineTo(x, y);
        }
    }

    // Fill under curve
    QPainterPath fillPath = path;
    fillPath.lineTo(2 + w, 2 + h);
    fillPath.lineTo(2, 2 + h);
    fillPath.closeSubpath();
    p.fillPath(fillPath, COLOR_CHART_FILL);

    // Draw line
    p.setPen(QPen(COLOR_CHART_LINE, 2));
    p.drawPath(path);

    // Draw 100% line
    p.setPen(QPen(COLOR_SUCCESS, 1, Qt::DashLine));
    p.drawLine(2, 2, 2 + w, 2);
}

// ========== MetricsPanel ==========

MetricsPanel::MetricsPanel(ZitEngine* engine, QWidget* parent)
    : QWidget(parent)
    , m_engine(engine)
{
    setStyleSheet(QString(
        "QLabel { color: white; font-family: 'SF Pro Display', 'Segoe UI', sans-serif; }"
        "QLabel#header { font-size: 10px; color: #808080; text-transform: uppercase; }"
        "QLabel#value { font-size: 18px; font-weight: 500; }"
        "QLabel#status { font-size: 12px; color: #00d4aa; }"
    ));

    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(12);
    layout->setContentsMargins(16, 16, 16, 16);

    // Cycle
    auto* cycleHeader = new QLabel("CYCLE");
    cycleHeader->setObjectName("header");
    m_cycleLabel = new QLabel("0");
    m_cycleLabel->setObjectName("value");

    // Resonant
    auto* resonantHeader = new QLabel("RESONANT");
    resonantHeader->setObjectName("header");
    m_resonantLabel = new QLabel("0 / 0");
    m_resonantLabel->setObjectName("value");

    // Rewires
    auto* rewiresHeader = new QLabel("REWIRES");
    rewiresHeader->setObjectName("header");
    m_rewiresLabel = new QLabel("0");
    m_rewiresLabel->setObjectName("value");

    // Status
    m_statusLabel = new QLabel("Ready");
    m_statusLabel->setObjectName("status");
    m_statusLabel->setAlignment(Qt::AlignCenter);

    // Chart
    auto* chartHeader = new QLabel("RESONANCE");
    chartHeader->setObjectName("header");
    m_chart = new ResonanceChart(this);

    // Add to layout
    layout->addWidget(cycleHeader);
    layout->addWidget(m_cycleLabel);
    layout->addSpacing(8);
    layout->addWidget(resonantHeader);
    layout->addWidget(m_resonantLabel);
    layout->addSpacing(8);
    layout->addWidget(rewiresHeader);
    layout->addWidget(m_rewiresLabel);
    layout->addSpacing(16);
    layout->addWidget(chartHeader);
    layout->addWidget(m_chart);
    layout->addStretch();
    layout->addWidget(m_statusLabel);

    // Connect to engine
    connect(engine, &ZitEngine::stepped, this, &MetricsPanel::updateMetrics);
    connect(engine, &ZitEngine::convergedSignal, this, [this]() {
        m_statusLabel->setText("CONVERGED");
        m_statusLabel->setStyleSheet("font-size: 12px; color: #00d4aa; font-weight: bold;");
    });
}

void MetricsPanel::updateMetrics() {
    m_cycleLabel->setText(QString::number(m_engine->cycle()));
    m_resonantLabel->setText(QString("%1 / %2")
        .arg(m_engine->resonant())
        .arg(m_engine->total()));
    m_rewiresLabel->setText(QString::number(m_engine->rewires()));

    if (m_engine->isRunning()) {
        m_statusLabel->setText("Running...");
        m_statusLabel->setStyleSheet("font-size: 12px; color: #1e90ff;");
    }

    m_chart->addDataPoint(m_engine->cycle(), m_engine->resonanceRatio());
}

void MetricsPanel::reset() {
    m_cycleLabel->setText("0");
    m_resonantLabel->setText(QString("0 / %1").arg(m_engine->total()));
    m_rewiresLabel->setText("0");
    m_statusLabel->setText("Ready");
    m_statusLabel->setStyleSheet("font-size: 12px; color: #808080;");
    m_chart->reset();
}
