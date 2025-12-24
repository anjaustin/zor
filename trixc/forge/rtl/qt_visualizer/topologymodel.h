// =============================================================================
// TOPOLOGYMODEL.H - Sacred Topology Data Model
// =============================================================================
//
// "Each node a star. Each edge a thread of meaning."
//
// C++ model for Qt Quick visualization of the learned topology.
// Bridges Verilog simulation output to QML visual components.
//
// =============================================================================

#ifndef TOPOLOGYMODEL_H
#define TOPOLOGYMODEL_H

#include <QObject>
#include <QAbstractListModel>
#include <QVariantList>
#include <QString>
#include <QTimer>
#include <QFile>
#include <QTextStream>
#include <array>
#include <vector>

// =============================================================================
// NODE DATA STRUCTURE
// =============================================================================

struct NodeData {
    int id;                           // 0-63
    std::array<int, 6> neighbors;     // Current neighbor indices
    std::array<int, 6> torus;         // Original torus neighbors
    int changeCount;                  // How many neighbors changed
    bool isResonant;                  // Currently resonant?
    int frustration;                  // Frustration level

    // Position in 3D torus
    int x() const { return id % 4; }
    int y() const { return (id / 4) % 4; }
    int z() const { return id / 16; }

    // Calculate torus neighbors
    void initTorus() {
        int px = x(), py = y(), pz = z();
        torus[0] = ((px + 1) % 4) + py * 4 + pz * 16;  // +X
        torus[1] = ((px + 3) % 4) + py * 4 + pz * 16;  // -X
        torus[2] = px + ((py + 1) % 4) * 4 + pz * 16;  // +Y
        torus[3] = px + ((py + 3) % 4) * 4 + pz * 16;  // -Y
        torus[4] = px + py * 4 + ((pz + 1) % 4) * 16;  // +Z
        torus[5] = px + py * 4 + ((pz + 3) % 4) * 16;  // -Z
    }

    // Count changed neighbors
    void updateChangeCount() {
        changeCount = 0;
        for (int i = 0; i < 6; i++) {
            if (neighbors[i] != torus[i]) changeCount++;
        }
    }
};

// =============================================================================
// CYCLE DATA (snapshot in time)
// =============================================================================

struct CycleData {
    int cycleNumber;
    std::array<NodeData, 64> nodes;
    int resonantCount;
    int globalFrustration;
    int rewiringCount;
};

// =============================================================================
// TOPOLOGY MODEL
// =============================================================================

class TopologyModel : public QAbstractListModel
{
    Q_OBJECT

    Q_PROPERTY(int currentCycle READ currentCycle WRITE setCurrentCycle NOTIFY currentCycleChanged)
    Q_PROPERTY(int totalCycles READ totalCycles NOTIFY totalCyclesChanged)
    Q_PROPERTY(int resonantCount READ resonantCount NOTIFY metricsChanged)
    Q_PROPERTY(int globalFrustration READ globalFrustration NOTIFY metricsChanged)
    Q_PROPERTY(int rewiringCount READ rewiringCount NOTIFY metricsChanged)
    Q_PROPERTY(bool isPlaying READ isPlaying NOTIFY playStateChanged)
    Q_PROPERTY(int playbackSpeed READ playbackSpeed WRITE setPlaybackSpeed NOTIFY playbackSpeedChanged)

public:
    enum Roles {
        IdRole = Qt::UserRole + 1,
        XRole,
        YRole,
        ZRole,
        NeighborsRole,
        TorusRole,
        ChangeCountRole,
        IsResonantRole,
        FrustrationRole,
        ColorRole
    };

    explicit TopologyModel(QObject *parent = nullptr);

    // QAbstractListModel interface
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    // Properties
    int currentCycle() const { return m_currentCycle; }
    int totalCycles() const { return static_cast<int>(m_cycles.size()); }
    int resonantCount() const;
    int globalFrustration() const;
    int rewiringCount() const;
    bool isPlaying() const { return m_playing; }
    int playbackSpeed() const { return m_playbackSpeed; }

    void setCurrentCycle(int cycle);
    void setPlaybackSpeed(int speed);

public slots:
    void loadFromFile(const QString &filename);
    void loadSampleData();
    void play();
    void pause();
    void stepForward();
    void stepBackward();
    void goToStart();
    void goToEnd();

    // Get edge data for drawing
    QVariantList getEdges(int layer = -1) const;
    QVariantList getNodesForLayer(int layer) const;

signals:
    void currentCycleChanged();
    void totalCyclesChanged();
    void metricsChanged();
    void playStateChanged();
    void playbackSpeedChanged();
    void dataLoaded();

private slots:
    void advancePlayback();

private:
    void parseLine(const QString &line);
    QString nodeColor(const NodeData &node) const;

    std::vector<CycleData> m_cycles;
    int m_currentCycle = 0;
    bool m_playing = false;
    int m_playbackSpeed = 100;  // ms between frames
    QTimer *m_playTimer;
};

#endif // TOPOLOGYMODEL_H
