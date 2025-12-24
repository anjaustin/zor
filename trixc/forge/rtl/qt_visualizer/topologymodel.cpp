// =============================================================================
// TOPOLOGYMODEL.CPP - Sacred Topology Data Model Implementation
// =============================================================================

#include "topologymodel.h"
#include <QDebug>
#include <QRegularExpression>

// =============================================================================
// SACRED COLORS
// =============================================================================

namespace Colors {
    const QString LayerBlue    = "#4a9eff";
    const QString LayerPurple  = "#b464ff";
    const QString LayerGreen   = "#64ff96";
    const QString LayerGold    = "#ffd700";
    const QString Magenta      = "#ff64c8";
    const QString DarkEdge     = "#444466";
    const QString Background   = "#1a1a2e";
}

// =============================================================================
// CONSTRUCTOR
// =============================================================================

TopologyModel::TopologyModel(QObject *parent)
    : QAbstractListModel(parent)
    , m_playTimer(new QTimer(this))
{
    connect(m_playTimer, &QTimer::timeout, this, &TopologyModel::advancePlayback);
}

// =============================================================================
// LIST MODEL INTERFACE
// =============================================================================

int TopologyModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid() || m_cycles.empty())
        return 0;
    return 64;
}

QVariant TopologyModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || m_cycles.empty())
        return QVariant();

    const auto &node = m_cycles[m_currentCycle].nodes[index.row()];

    switch (role) {
    case IdRole:
        return node.id;
    case XRole:
        return node.x();
    case YRole:
        return node.y();
    case ZRole:
        return node.z();
    case NeighborsRole: {
        QVariantList list;
        for (int n : node.neighbors)
            list.append(n);
        return list;
    }
    case TorusRole: {
        QVariantList list;
        for (int n : node.torus)
            list.append(n);
        return list;
    }
    case ChangeCountRole:
        return node.changeCount;
    case IsResonantRole:
        return node.isResonant;
    case FrustrationRole:
        return node.frustration;
    case ColorRole:
        return nodeColor(node);
    default:
        return QVariant();
    }
}

QHash<int, QByteArray> TopologyModel::roleNames() const
{
    return {
        {IdRole, "nodeId"},
        {XRole, "nodeX"},
        {YRole, "nodeY"},
        {ZRole, "nodeZ"},
        {NeighborsRole, "neighbors"},
        {TorusRole, "torusNeighbors"},
        {ChangeCountRole, "changeCount"},
        {IsResonantRole, "isResonant"},
        {FrustrationRole, "frustration"},
        {ColorRole, "nodeColor"}
    };
}

// =============================================================================
// PROPERTIES
// =============================================================================

int TopologyModel::resonantCount() const
{
    if (m_cycles.empty()) return 0;
    return m_cycles[m_currentCycle].resonantCount;
}

int TopologyModel::globalFrustration() const
{
    if (m_cycles.empty()) return 0;
    return m_cycles[m_currentCycle].globalFrustration;
}

int TopologyModel::rewiringCount() const
{
    if (m_cycles.empty()) return 0;
    return m_cycles[m_currentCycle].rewiringCount;
}

void TopologyModel::setCurrentCycle(int cycle)
{
    if (cycle < 0) cycle = 0;
    if (cycle >= static_cast<int>(m_cycles.size()))
        cycle = m_cycles.size() - 1;

    if (m_currentCycle != cycle) {
        m_currentCycle = cycle;
        emit currentCycleChanged();
        emit metricsChanged();
        emit dataChanged(index(0), index(63));
    }
}

void TopologyModel::setPlaybackSpeed(int speed)
{
    if (m_playbackSpeed != speed) {
        m_playbackSpeed = speed;
        if (m_playing) {
            m_playTimer->setInterval(speed);
        }
        emit playbackSpeedChanged();
    }
}

// =============================================================================
// PLAYBACK CONTROL
// =============================================================================

void TopologyModel::play()
{
    if (!m_playing && !m_cycles.empty()) {
        m_playing = true;
        m_playTimer->start(m_playbackSpeed);
        emit playStateChanged();
    }
}

void TopologyModel::pause()
{
    if (m_playing) {
        m_playing = false;
        m_playTimer->stop();
        emit playStateChanged();
    }
}

void TopologyModel::stepForward()
{
    setCurrentCycle(m_currentCycle + 1);
}

void TopologyModel::stepBackward()
{
    setCurrentCycle(m_currentCycle - 1);
}

void TopologyModel::goToStart()
{
    setCurrentCycle(0);
}

void TopologyModel::goToEnd()
{
    setCurrentCycle(m_cycles.size() - 1);
}

void TopologyModel::advancePlayback()
{
    if (m_currentCycle < static_cast<int>(m_cycles.size()) - 1) {
        setCurrentCycle(m_currentCycle + 1);
    } else {
        pause();
    }
}

// =============================================================================
// DATA LOADING
// =============================================================================

void TopologyModel::loadFromFile(const QString &filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Could not open file:" << filename;
        return;
    }

    m_cycles.clear();
    CycleData currentCycle;
    currentCycle.cycleNumber = 0;

    // Initialize all nodes with torus topology
    for (int i = 0; i < 64; i++) {
        currentCycle.nodes[i].id = i;
        currentCycle.nodes[i].initTorus();
        currentCycle.nodes[i].neighbors = currentCycle.nodes[i].torus;
        currentCycle.nodes[i].changeCount = 0;
        currentCycle.nodes[i].isResonant = false;
        currentCycle.nodes[i].frustration = 0;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        parseLine(in.readLine());
    }

    if (!m_cycles.empty()) {
        m_currentCycle = 0;
        beginResetModel();
        endResetModel();
        emit totalCyclesChanged();
        emit currentCycleChanged();
        emit metricsChanged();
        emit dataLoaded();
    }
}

void TopologyModel::parseLine(const QString &line)
{
    if (line.startsWith('#') || line.isEmpty())
        return;

    QStringList parts = line.split(',');

    if (parts[0] == "TOPO" && parts.size() >= 9) {
        // TOPO,cycle,node,n0,n1,n2,n3,n4,n5
        int cycle = parts[1].toInt();
        int nodeId = parts[2].toInt();

        // Ensure we have this cycle
        while (m_cycles.size() <= static_cast<size_t>(cycle)) {
            CycleData newCycle;
            newCycle.cycleNumber = m_cycles.size();
            for (int i = 0; i < 64; i++) {
                newCycle.nodes[i].id = i;
                newCycle.nodes[i].initTorus();
                if (!m_cycles.empty()) {
                    newCycle.nodes[i] = m_cycles.back().nodes[i];
                } else {
                    newCycle.nodes[i].neighbors = newCycle.nodes[i].torus;
                }
            }
            m_cycles.push_back(newCycle);
        }

        // Update node neighbors
        for (int i = 0; i < 6; i++) {
            m_cycles[cycle].nodes[nodeId].neighbors[i] = parts[3 + i].toInt();
        }
        m_cycles[cycle].nodes[nodeId].updateChangeCount();
    }
    else if (parts[0] == "METRICS" && parts.size() >= 5) {
        // METRICS,cycle,resonant,frustration,rewiring
        int cycle = parts[1].toInt();
        if (cycle < static_cast<int>(m_cycles.size())) {
            m_cycles[cycle].resonantCount = parts[2].toInt();
            m_cycles[cycle].globalFrustration = parts[3].toInt();
            m_cycles[cycle].rewiringCount = parts[4].toInt();
        }
    }
}

void TopologyModel::loadSampleData()
{
    // Create sample learned topology for demo
    m_cycles.clear();

    CycleData cycle;
    cycle.cycleNumber = 0;
    cycle.resonantCount = 64;
    cycle.globalFrustration = 0;
    cycle.rewiringCount = 5;

    for (int i = 0; i < 64; i++) {
        cycle.nodes[i].id = i;
        cycle.nodes[i].initTorus();
        cycle.nodes[i].neighbors = cycle.nodes[i].torus;
        cycle.nodes[i].isResonant = true;
        cycle.nodes[i].frustration = 0;
    }

    // Add learned edges (from actual experiment)
    cycle.nodes[1].neighbors = {31, 62, 63, 13, 17, 49};
    cycle.nodes[32].neighbors = {10, 35, 36, 44, 48, 16};
    cycle.nodes[63].neighbors = {46, 58, 37, 24, 33, 47};

    for (int i = 0; i < 64; i++) {
        cycle.nodes[i].updateChangeCount();
    }

    m_cycles.push_back(cycle);

    m_currentCycle = 0;
    beginResetModel();
    endResetModel();
    emit totalCyclesChanged();
    emit currentCycleChanged();
    emit metricsChanged();
    emit dataLoaded();
}

// =============================================================================
// EDGE DATA FOR DRAWING
// =============================================================================

QVariantList TopologyModel::getEdges(int layer) const
{
    QVariantList edges;
    if (m_cycles.empty()) return edges;

    const auto &cycle = m_cycles[m_currentCycle];
    QSet<QPair<int,int>> drawn;

    for (int i = 0; i < 64; i++) {
        const auto &node = cycle.nodes[i];

        // Filter by layer if specified
        if (layer >= 0 && node.z() != layer)
            continue;

        for (int j = 0; j < 6; j++) {
            int neighbor = node.neighbors[j];

            // Filter edges to same layer if specified
            if (layer >= 0 && (neighbor / 16) != layer)
                continue;

            // Avoid duplicates
            int a = qMin(i, neighbor);
            int b = qMax(i, neighbor);
            if (drawn.contains({a, b}))
                continue;
            drawn.insert({a, b});

            QVariantMap edge;
            edge["from"] = i;
            edge["to"] = neighbor;
            edge["isLearned"] = (node.neighbors[j] != node.torus[j]);
            edges.append(edge);
        }
    }

    return edges;
}

QVariantList TopologyModel::getNodesForLayer(int layer) const
{
    QVariantList nodes;
    if (m_cycles.empty()) return nodes;

    const auto &cycle = m_cycles[m_currentCycle];

    for (int i = 0; i < 64; i++) {
        if (cycle.nodes[i].z() == layer) {
            QVariantMap node;
            node["id"] = i;
            node["x"] = cycle.nodes[i].x();
            node["y"] = cycle.nodes[i].y();
            node["changeCount"] = cycle.nodes[i].changeCount;
            node["color"] = nodeColor(cycle.nodes[i]);
            nodes.append(node);
        }
    }

    return nodes;
}

// =============================================================================
// HELPERS
// =============================================================================

QString TopologyModel::nodeColor(const NodeData &node) const
{
    static const QString layerColors[] = {
        Colors::LayerBlue,
        Colors::LayerPurple,
        Colors::LayerGreen,
        Colors::LayerGold
    };

    if (node.changeCount == 0) {
        return layerColors[node.z()];
    } else if (node.changeCount < 3) {
        return Colors::Magenta;
    } else {
        return Colors::LayerGold;
    }
}
