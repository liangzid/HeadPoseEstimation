import sys
from PyQt5.QtWidgets import QApplication ,QMainWindow
from assist_expression import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage , QPixmap 
import test_on_video_dlib_new
import cv2
import moveAndEnter_new
import time

class MyMainWindows(QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(MyMainWindows,self).__init__(parent)
        self.setupUi(self)
        # self.QTimer_camera = QTimer()
        # self.QTimer_camera.start(50)
        # self.QTimer_camera.timeout.connect(self.openFrame)

    def openFrame(self):
        # global test_on_video_dlib_new.frame  
        if test_on_video_dlib_new.frame is not None:
            test_on_video_dlib_new.frame = cv2.cvtColor(test_on_video_dlib_new.frame,cv2.COLOR_BGR2RGB)
            height , width ,bytesperComponet = test_on_video_dlib_new.frame.shape
            bytesperline = bytesperComponet * width
            q_image = QImage(test_on_video_dlib_new.frame.data,  width, height, bytesperline, 
                QImage.Format_RGB888).scaled(self.dispalylabel.width(), self.dispalylabel.height())
            self.dispalylabel.setPixmap(QPixmap.fromImage(q_image)) 

class Workthread(QThread):
    trigger = pyqtSignal()
    def __init__(self,myMin):
        super(Workthread,self).__init__()
        self.trigger.connect(myMin.openFrame)
    def run(self):
        test_on_video_dlib_new.test_on_video_dlib_new_init(self.trigger)
        # print([test_on_video_dlib_new.yaw_predicted , test_on_video_dlib_new.pitch_predict , test_on_video_dlib_new.roll_predict])

class Movethread(QThread):
    def __init__(self):
        super(Movethread,self).__init__()   # -2 2 -7 
        self.keyboardmove=moveAndEnter_new.MoveAndEnter(0.4,0.4,[0,-3,-1])
    def run(self):
        while(1):
            time.sleep(0.02)
            if test_on_video_dlib_new.yawx is not None:
                #print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",float(test_on_video_dlib_new.yawx.data))
                self.keyboardmove.movee([float(test_on_video_dlib_new.yawx.data) , float(test_on_video_dlib_new.pitchx.data) , 
                    float(test_on_video_dlib_new.rollx.data)])



if __name__ == "__main__":

    app = QApplication(sys.argv)
    myMin = MyMainWindows()
    Workthread = Workthread(myMin)
    Movethread=Movethread()
    myMin.show()
    Workthread.start()
    Movethread.start()
    sys.exit(app.exec_())
