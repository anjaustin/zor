#ifndef CONTROLPANEL_H
#define CONTROLPANEL_H

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QCheckBox>
#include <QButtonGroup>
#include <QRadioButton>

class ZitEngine;
class FabricRenderer;

class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(ZitEngine* engine, FabricRenderer* renderer,
                          QWidget* parent = nullptr);

signals:
    void resetRequested();

private slots:
    void onPlayPause();
    void onStep();
    void onReset();
    void onSpeedChanged(int value);
    void onColorModeChanged(int id);
    void onShowEdgesChanged(bool checked);

private:
    void updatePlayButton();

    ZitEngine* m_engine;
    FabricRenderer* m_renderer;

    QPushButton* m_playButton;
    QPushButton* m_stepButton;
    QPushButton* m_resetButton;
    QSlider* m_speedSlider;
    QButtonGroup* m_colorModeGroup;
    QCheckBox* m_showEdgesCheck;
};

#endif // CONTROLPANEL_H
