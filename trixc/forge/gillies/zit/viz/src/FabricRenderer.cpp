#include "FabricRenderer.h"
#include "ZitEngine.h"
#include <QtMath>
#include <QColor>

// BMW-inspired color palette
static const QColor COLOR_BACKGROUND(10, 10, 15);
static const QColor COLOR_RESONANT(0, 212, 170);      // Teal - stable
static const QColor COLOR_LOW_RESIST(68, 136, 255);   // Blue - slight strain
static const QColor COLOR_MID_RESIST(136, 68, 255);   // Purple - building
static const QColor COLOR_HIGH_RESIST(255, 165, 0);   // Orange - near threshold
static const QColor COLOR_REWIRING(255, 68, 68);      // Red - rewiring
static const QColor COLOR_EDGE(100, 100, 120, 77);    // Gray with alpha

FabricRenderer::FabricRenderer(ZitEngine* engine, QWidget* parent)
    : QOpenGLWidget(parent)
    , m_engine(engine)
    , m_nodeShader(nullptr)
    , m_edgeShader(nullptr)
    , m_sphereIndexCount(0)
    , m_rotX(30.0f)
    , m_rotY(45.0f)
    , m_zoom(15.0f)
    , m_colorMode(ByResonance)
    , m_showEdges(true)
    , m_showRewires(true)
{
    connect(engine, &ZitEngine::stepped, this, QOverload<>::of(&QOpenGLWidget::update));
}

FabricRenderer::~FabricRenderer() {
    makeCurrent();
    delete m_nodeShader;
    delete m_edgeShader;
    m_sphereVBO.destroy();
    m_sphereIBO.destroy();
    m_sphereVAO.destroy();
    m_edgeVBO.destroy();
    m_edgeVAO.destroy();
    doneCurrent();
}

void FabricRenderer::setColorMode(ColorMode mode) {
    m_colorMode = mode;
    update();
}

void FabricRenderer::setShowEdges(bool show) {
    m_showEdges = show;
    update();
}

void FabricRenderer::setShowRewires(bool show) {
    m_showRewires = show;
    update();
}

void FabricRenderer::resetCamera() {
    m_rotX = 30.0f;
    m_rotY = 45.0f;
    m_zoom = 15.0f;
    update();
}

void FabricRenderer::initializeGL() {
    initializeOpenGLFunctions();

    glClearColor(COLOR_BACKGROUND.redF(), COLOR_BACKGROUND.greenF(),
                 COLOR_BACKGROUND.blueF(), 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    setupShaders();
    setupGeometry();
}

void FabricRenderer::setupShaders() {
    // Node shader (simple Phong-like shading)
    m_nodeShader = new QOpenGLShaderProgram(this);
    m_nodeShader->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 nodeColor;

        out vec3 fragNormal;
        out vec3 fragColor;

        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
            fragNormal = mat3(model) * normal;
            fragColor = nodeColor;
        }
    )");
    m_nodeShader->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        #version 330 core
        in vec3 fragNormal;
        in vec3 fragColor;
        out vec4 color;

        void main() {
            vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
            float ambient = 0.3;
            float diffuse = max(dot(normalize(fragNormal), lightDir), 0.0) * 0.7;
            vec3 lit = fragColor * (ambient + diffuse);
            color = vec4(lit, 1.0);
        }
    )");
    m_nodeShader->link();

    // Edge shader (simple lines)
    m_edgeShader = new QOpenGLShaderProgram(this);
    m_edgeShader->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        #version 330 core
        layout(location = 0) in vec3 position;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * vec4(position, 1.0);
        }
    )");
    m_edgeShader->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        #version 330 core
        uniform vec4 lineColor;
        out vec4 color;

        void main() {
            color = lineColor;
        }
    )");
    m_edgeShader->link();
}

void FabricRenderer::setupGeometry() {
    // Generate sphere geometry (icosphere approximation)
    const int rings = 12;
    const int sectors = 24;
    const float radius = 0.15f;

    QVector<float> vertices;
    QVector<unsigned int> indices;

    for (int r = 0; r <= rings; ++r) {
        float phi = M_PI * float(r) / float(rings);
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * M_PI * float(s) / float(sectors);

            float x = radius * sin(phi) * cos(theta);
            float y = radius * cos(phi);
            float z = radius * sin(phi) * sin(theta);

            // Position
            vertices << x << y << z;
            // Normal (normalized position for sphere)
            vertices << x/radius << y/radius << z/radius;
        }
    }

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            int first = r * (sectors + 1) + s;
            int second = first + sectors + 1;

            indices << first << second << first + 1;
            indices << second << second + 1 << first + 1;
        }
    }

    m_sphereIndexCount = indices.size();

    m_sphereVAO.create();
    m_sphereVAO.bind();

    m_sphereVBO.create();
    m_sphereVBO.bind();
    m_sphereVBO.allocate(vertices.constData(), vertices.size() * sizeof(float));

    m_sphereIBO.create();
    m_sphereIBO.bind();
    m_sphereIBO.allocate(indices.constData(), indices.size() * sizeof(unsigned int));

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    m_sphereVAO.release();

    // Edge VAO (populated dynamically)
    m_edgeVAO.create();
    m_edgeVBO.create();
}

void FabricRenderer::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_showEdges) {
        renderEdges();
    }

    renderNodes();
}

void FabricRenderer::renderNodes() {
    if (!m_nodeShader) return;

    m_nodeShader->bind();
    m_nodeShader->setUniformValue("view", viewMatrix());
    m_nodeShader->setUniformValue("projection", projectionMatrix());

    m_sphereVAO.bind();
    m_sphereIBO.bind();

    int total = m_engine->total();
    for (int i = 0; i < total; ++i) {
        QVector3D pos = m_engine->nodePosition(i);

        QMatrix4x4 model;
        model.translate(pos);

        m_nodeShader->setUniformValue("model", model);

        QColor c = nodeColor(i);
        m_nodeShader->setUniformValue("nodeColor",
            QVector3D(c.redF(), c.greenF(), c.blueF()));

        glDrawElements(GL_TRIANGLES, m_sphereIndexCount, GL_UNSIGNED_INT, nullptr);
    }

    m_sphereVAO.release();
    m_nodeShader->release();
}

void FabricRenderer::renderEdges() {
    // Build edge data
    QVector<float> edgeVertices;
    int total = m_engine->total();

    for (int i = 0; i < total; ++i) {
        QVector3D pos = m_engine->nodePosition(i);

        // Only draw edges in +X, +Y, +Z directions to avoid duplicates
        for (int d = 0; d < 6; d += 2) {
            int neighbor = m_engine->nodeNeighbor(i, d);
            if (neighbor > i) {  // Only draw each edge once
                QVector3D nPos = m_engine->nodePosition(neighbor);
                edgeVertices << pos.x() << pos.y() << pos.z();
                edgeVertices << nPos.x() << nPos.y() << nPos.z();
            }
        }
    }

    if (edgeVertices.isEmpty()) return;

    m_edgeVAO.bind();
    m_edgeVBO.bind();
    m_edgeVBO.allocate(edgeVertices.constData(), edgeVertices.size() * sizeof(float));

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    m_edgeShader->bind();
    m_edgeShader->setUniformValue("view", viewMatrix());
    m_edgeShader->setUniformValue("projection", projectionMatrix());
    m_edgeShader->setUniformValue("lineColor",
        QVector4D(COLOR_EDGE.redF(), COLOR_EDGE.greenF(),
                  COLOR_EDGE.blueF(), COLOR_EDGE.alphaF()));

    glDrawArrays(GL_LINES, 0, edgeVertices.size() / 3);

    m_edgeShader->release();
    m_edgeVAO.release();
}

QColor FabricRenderer::nodeColor(int nodeId) const {
    if (m_engine->nodeRewiring(nodeId)) {
        return COLOR_REWIRING;
    }

    switch (m_colorMode) {
        case ByResonance:
            return m_engine->nodeResonant(nodeId) ? COLOR_RESONANT : COLOR_HIGH_RESIST;

        case ByResistance: {
            uint8_t r = m_engine->nodeResistance(nodeId);
            if (r == 0) return COLOR_RESONANT;
            if (r < 3) return COLOR_LOW_RESIST;
            if (r < 6) return COLOR_MID_RESIST;
            return COLOR_HIGH_RESIST;
        }

        case ByState: {
            uint8_t s = m_engine->nodeState(nodeId);
            // Map 0-255 to hue 0-360
            float hue = float(s) / 255.0f * 360.0f;
            return QColor::fromHsvF(hue / 360.0f, 0.8f, 0.9f);
        }
    }

    return COLOR_RESONANT;
}

QMatrix4x4 FabricRenderer::viewMatrix() const {
    QMatrix4x4 view;
    view.translate(0, 0, -m_zoom);
    view.rotate(m_rotX, 1, 0, 0);
    view.rotate(m_rotY, 0, 1, 0);
    return view;
}

QMatrix4x4 FabricRenderer::projectionMatrix() const {
    QMatrix4x4 projection;
    float aspect = float(width()) / float(height());
    projection.perspective(45.0f, aspect, 0.1f, 100.0f);
    return projection;
}

void FabricRenderer::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void FabricRenderer::mousePressEvent(QMouseEvent* event) {
    m_lastMousePos = event->pos();
}

void FabricRenderer::mouseMoveEvent(QMouseEvent* event) {
    int dx = event->pos().x() - m_lastMousePos.x();
    int dy = event->pos().y() - m_lastMousePos.y();

    if (event->buttons() & Qt::LeftButton) {
        m_rotY += dx * 0.5f;
        m_rotX += dy * 0.5f;
        m_rotX = qBound(-90.0f, m_rotX, 90.0f);
        update();
    }

    m_lastMousePos = event->pos();
}

void FabricRenderer::wheelEvent(QWheelEvent* event) {
    float delta = event->angleDelta().y() / 120.0f;
    m_zoom -= delta;
    m_zoom = qBound(5.0f, m_zoom, 50.0f);
    update();
}
