from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QGraphicsView


class ImageView(QGraphicsView):
    def __init__(self, centralwidget):
        super(ImageView, self).__init__(centralwidget)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   
        self.setDragMode(QGraphicsView.ScrollHandDrag)          

    def wheelEvent(self, event: QWheelEvent) -> None:
        if len(self.scene().items()) == 0:
            return

        curPoint = event.position()
        scenePos = self.mapToScene(QPoint(curPoint.x(), curPoint.y()))
    
        viewWidth = self.viewport().width()
        viewHeight = self.viewport().height()
    
        hScale = curPoint.x() / viewWidth
        vScale = curPoint.y() / viewHeight
    
        wheelDeltaValue = event.angleDelta().y()
        scaleFactor = self.transform().m11()
        if (scaleFactor < 0.05 and wheelDeltaValue<0) or (scaleFactor>50 and wheelDeltaValue>0):
            return

        if wheelDeltaValue > 0:
            self.scale(1.2, 1.2)
        else:
            self.scale(1.0/1.2, 1.0/1.2)
        
        viewPoint = self.transform().map(scenePos)
        self.horizontalScrollBar().setValue(int(viewPoint.x() - viewWidth * hScale ))
        self.verticalScrollBar().setValue(int(viewPoint.y() - viewHeight * vScale ))
    
        self.update()
