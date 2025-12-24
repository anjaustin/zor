// =============================================================================
// MAIN.CPP - Sacred Topology Visualizer Entry Point
// =============================================================================
//
// "The topology IS the self. Now we SEE the self."
//
// Qt6 application for visualizing self-organizing topological fabrics.
//
// =============================================================================

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include "topologymodel.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setApplicationName("Sacred Topology Visualizer");
    app.setOrganizationName("TrixO");
    app.setApplicationVersion("1.0");

    // Set Material style for modern look
    QQuickStyle::setStyle("Material");

    // Create the topology model
    TopologyModel model;

    // Load data from command line or use sample
    if (argc > 1) {
        model.loadFromFile(QString::fromLocal8Bit(argv[1]));
    } else {
        model.loadSampleData();
    }

    // Create QML engine
    QQmlApplicationEngine engine;

    // Expose model to QML
    engine.rootContext()->setContextProperty("topologyModel", &model);

    // Load main QML file
    const QUrl url(QStringLiteral("qrc:/Main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}
