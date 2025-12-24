#ifndef METRICSPANEL_H
#define METRICSPANEL_H

#include <QWidget>
#include <QLabel>
#include <QVector>
#include <QPointF>

class ZitEngine;

class ResonanceChart : public QWidget {
    Q_OBJECT

public:
    explicit ResonanceChart(QWidget* parent = nullptr);

    void addDataPoint(int cycle, float resonanceRatio);
    void reset();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QVector<QPointF> m_data;
    static const int MAX_POINTS = 500;
};

class MetricsPanel : public QWidget {
    Q_OBJECT

public:
    explicit MetricsPanel(ZitEngine* engine, QWidget* parent = nullptr);

public slots:
    void updateMetrics();
    void reset();

private:
    ZitEngine* m_engine;

    QLabel* m_cycleLabel;
    QLabel* m_resonantLabel;
    QLabel* m_rewiresLabel;
    QLabel* m_statusLabel;
    ResonanceChart* m_chart;
};

#endif // METRICSPANEL_H
