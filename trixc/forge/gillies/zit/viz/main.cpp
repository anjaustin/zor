/*
 * ZIT Fabric Visualizer
 *
 * Real-time visualization of homeo-adaptive topological learning.
 * Watch 512 nodes discover their optimal connectivity through
 * resistance and rewiring.
 *
 * The topology IS the learned model.
 */

#include <QApplication>
#include <QSurfaceFormat>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    // Request OpenGL 3.3 Core Profile
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSamples(4);  // Antialiasing
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);

    // Application metadata
    app.setApplicationName("ZIT Fabric Visualizer");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("ZOR Project");

    // Dark palette for native widgets
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(10, 10, 15));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(20, 20, 26));
    darkPalette.setColor(QPalette::AlternateBase, QColor(30, 30, 40));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(30, 30, 40));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, QColor(30, 144, 255));
    darkPalette.setColor(QPalette::Highlight, QColor(30, 144, 255));
    darkPalette.setColor(QPalette::HighlightedText, Qt::white);
    app.setPalette(darkPalette);

    MainWindow window;
    window.show();

    return app.exec();
}
