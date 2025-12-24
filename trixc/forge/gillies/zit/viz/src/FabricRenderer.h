#ifndef FABRICRENDERER_H
#define FABRICRENDERER_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QMatrix4x4>
#include <QVector3D>
#include <QMouseEvent>
#include <QWheelEvent>

class ZitEngine;

class FabricRenderer : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    enum ColorMode {
        ByState,
        ByResistance,
        ByResonance
    };

    explicit FabricRenderer(ZitEngine* engine, QWidget* parent = nullptr);
    ~FabricRenderer();

    void setColorMode(ColorMode mode);
    ColorMode colorMode() const { return m_colorMode; }

    void setShowEdges(bool show);
    bool showEdges() const { return m_showEdges; }

    void setShowRewires(bool show);
    bool showRewires() const { return m_showRewires; }

    void resetCamera();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void setupShaders();
    void setupGeometry();
    void renderNodes();
    void renderEdges();
    QColor nodeColor(int nodeId) const;
    QMatrix4x4 viewMatrix() const;
    QMatrix4x4 projectionMatrix() const;

    ZitEngine* m_engine;

    // Shaders
    QOpenGLShaderProgram* m_nodeShader;
    QOpenGLShaderProgram* m_edgeShader;

    // Geometry for instanced sphere rendering
    QOpenGLVertexArrayObject m_sphereVAO;
    QOpenGLBuffer m_sphereVBO;
    QOpenGLBuffer m_sphereIBO{QOpenGLBuffer::IndexBuffer};
    int m_sphereIndexCount;

    // Geometry for edges
    QOpenGLVertexArrayObject m_edgeVAO;
    QOpenGLBuffer m_edgeVBO;

    // Camera
    float m_rotX;
    float m_rotY;
    float m_zoom;
    QPoint m_lastMousePos;

    // State
    ColorMode m_colorMode;
    bool m_showEdges;
    bool m_showRewires;
};

#endif // FABRICRENDERER_H
