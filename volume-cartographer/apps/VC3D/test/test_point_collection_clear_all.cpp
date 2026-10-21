#include "CPointCollectionWidget.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QApplication>
#include <QPushButton>
#include <QTreeView>
#include <QtTest>

class TestPointCollectionClearAll : public QObject {
    Q_OBJECT
private slots:
    void clearAllLeavesNoStaleRows()
    {
        VCCollection collection;
        collection.addPoints("sheet_a", {{100, 200, 300}, {101, 201, 301}, {102, 202, 302}});
        collection.addPoints("sheet_b", {{400, 500, 600}, {401, 501, 601}});
        CPointCollectionWidget widget(&collection);
        widget.resize(480, 480);
        widget.show();
        QVERIFY(QTest::qWaitForWindowExposed(&widget));
        auto* tree = widget.findChild<QTreeView*>(QStringLiteral("pointCollectionTreeView"));
        auto* clearAll = widget.findChild<QPushButton*>(QStringLiteral("pointCollectionClearAllButton"));
        QVERIFY(tree && clearAll);
        tree->expandAll();
        QVERIFY(!widget.grab().isNull());

        clearAll->click();

        QCOMPARE(tree->model()->rowCount(), 0);
        // The view must have seen the reset; on main it still hit-tests a deleted row.
        QVERIFY(!tree->indexAt(QPoint(10, 10)).isValid());
        QVERIFY(!widget.grab().isNull());
    }
};

QTEST_MAIN(TestPointCollectionClearAll)
#include "test_point_collection_clear_all.moc"
