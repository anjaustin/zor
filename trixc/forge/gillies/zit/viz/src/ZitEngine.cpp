#include "ZitEngine.h"
#include <QDateTime>

ZitEngine::ZitEngine(int dim, QObject* parent)
    : QObject(parent)
    , m_fabric(nullptr)
    , m_dim(dim)
    , m_speed(1.0f)
    , m_running(false)
{
    m_fabric = zit_create(dim);
    zit_seed(m_fabric, 1122911624);  // Second Star Constant

    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &ZitEngine::onTimerTick);
}

ZitEngine::~ZitEngine() {
    if (m_fabric) {
        zit_destroy(m_fabric);
    }
}

void ZitEngine::step() {
    if (!m_fabric) return;

    zit_step(m_fabric);
    emit stepped();

    if (zit_converged(m_fabric)) {
        stop();
        emit convergedSignal();
    }
}

void ZitEngine::reset(uint32_t seed) {
    if (!m_fabric) return;

    bool wasRunning = m_running;
    stop();

    if (seed == 0) {
        seed = static_cast<uint32_t>(QDateTime::currentMSecsSinceEpoch() & 0xFFFFFFFF);
    }
    zit_seed(m_fabric, seed);
    emit stepped();

    if (wasRunning) {
        start();
    }
}

void ZitEngine::start() {
    if (m_running) return;
    if (converged()) return;

    m_running = true;
    int interval = static_cast<int>(50.0f / m_speed);  // Base: 20 steps/sec
    m_timer->start(interval);
    emit started();
}

void ZitEngine::stop() {
    if (!m_running) return;

    m_running = false;
    m_timer->stop();
    emit stopped();
}

bool ZitEngine::isRunning() const {
    return m_running;
}

void ZitEngine::setSpeed(float multiplier) {
    m_speed = qBound(0.1f, multiplier, 10.0f);
    if (m_running) {
        int interval = static_cast<int>(50.0f / m_speed);
        m_timer->setInterval(interval);
    }
}

int ZitEngine::dim() const {
    return m_dim;
}

int ZitEngine::total() const {
    return m_fabric ? zit_total(m_fabric) : 0;
}

int ZitEngine::cycle() const {
    return m_fabric ? zit_cycle(m_fabric) : 0;
}

int ZitEngine::resonant() const {
    return m_fabric ? zit_resonant(m_fabric) : 0;
}

int ZitEngine::rewires() const {
    return m_fabric ? zit_rewires(m_fabric) : 0;
}

bool ZitEngine::converged() const {
    return m_fabric ? zit_converged(m_fabric) : false;
}

float ZitEngine::resonanceRatio() const {
    if (!m_fabric) return 0.0f;
    int t = zit_total(m_fabric);
    if (t == 0) return 0.0f;
    return static_cast<float>(zit_resonant(m_fabric)) / static_cast<float>(t);
}

uint8_t ZitEngine::nodeState(int id) const {
    return m_fabric ? zit_node_state(m_fabric, id) : 0;
}

uint8_t ZitEngine::nodeResistance(int id) const {
    return m_fabric ? zit_node_resistance(m_fabric, id) : 0;
}

bool ZitEngine::nodeResonant(int id) const {
    return m_fabric ? zit_node_resonant(m_fabric, id) : false;
}

bool ZitEngine::nodeRewiring(int id) const {
    return m_fabric ? zit_node_rewiring(m_fabric, id) : false;
}

QVector3D ZitEngine::nodePosition(int id) const {
    if (!m_fabric) return QVector3D();

    int x, y, z;
    zit_node_coords(m_fabric, id, &x, &y, &z);

    // Center the fabric around origin
    float offset = static_cast<float>(m_dim - 1) / 2.0f;
    return QVector3D(
        static_cast<float>(x) - offset,
        static_cast<float>(y) - offset,
        static_cast<float>(z) - offset
    );
}

int ZitEngine::nodeNeighbor(int id, int direction) const {
    return m_fabric ? zit_node_neighbor(m_fabric, id, direction) : -1;
}

void ZitEngine::onTimerTick() {
    step();
}
