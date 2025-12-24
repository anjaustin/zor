#ifndef ZITENGINE_H
#define ZITENGINE_H

#include <QObject>
#include <QTimer>
#include <QVector3D>
#include <QList>

extern "C" {
#include "zit.h"
}

class ZitEngine : public QObject {
    Q_OBJECT

public:
    explicit ZitEngine(int dim = 8, QObject* parent = nullptr);
    ~ZitEngine();

    // Control
    void step();
    void reset(uint32_t seed = 0);
    void start();
    void stop();
    bool isRunning() const;
    void setSpeed(float multiplier);
    float speed() const { return m_speed; }

    // State queries
    int dim() const;
    int total() const;
    int cycle() const;
    int resonant() const;
    int rewires() const;
    bool converged() const;
    float resonanceRatio() const;

    // Per-node queries
    uint8_t nodeState(int id) const;
    uint8_t nodeResistance(int id) const;
    bool nodeResonant(int id) const;
    bool nodeRewiring(int id) const;
    QVector3D nodePosition(int id) const;
    int nodeNeighbor(int id, int direction) const;

signals:
    void stepped();
    void convergedSignal();
    void started();
    void stopped();

private slots:
    void onTimerTick();

private:
    zit_fabric_t* m_fabric;
    QTimer* m_timer;
    int m_dim;
    float m_speed;
    bool m_running;
};

#endif // ZITENGINE_H
