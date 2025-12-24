#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class ZitEngine;
class FabricRenderer;
class MetricsPanel;
class ControlPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;

protected:
    void keyPressEvent(QKeyEvent* event) override;

private:
    ZitEngine* m_engine;
    FabricRenderer* m_renderer;
    MetricsPanel* m_metrics;
    ControlPanel* m_controls;
};

#endif // MAINWINDOW_H
