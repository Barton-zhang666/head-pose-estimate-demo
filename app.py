import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from HPEqt.main import Ui_MainWindow
import sys
import numpy as np
from detect import get_model,get_pose,face_detection

class MyWindow(Ui_MainWindow):
    def initial(self):
        # 初始化 QGraphicsScene
        self.scene = QtWidgets.QGraphicsScene()
        self.videoshow.setScene(self.scene)

        # 连接按钮信号
        self.startButton.clicked.connect(self.load_image)
        self.stopButton.clicked.connect(self.clear_image)

        #加载模型
        self.device = "cuda"
        self.net,self.model = get_model(
            netpath = "./weight/Resnet50_Final.pth",
            modelpath = "./weight/6DRepNet_300W_LP_AFLW2000.pth",
            device = self.device
        )

        # 保存最近两秒的检测帧
        self.saved_frame_ola = -100

        # 设置音频文件路径
        self.audio_path = "resources/alarm.wav"  # 替换为你的音频文件路径
        # 创建播放器对象
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.audio_path)))

    def load_image(self):
        # 使用 OpenCV 读取图像
        self.cam = cv2.VideoCapture(0)

        # 检查摄像头是否成功打开
        if not self.cam.isOpened():
            print("无法打开摄像头")
            return

        # 创建一个定时器，用于定期更新图像
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 每 30 毫秒更新一次

        # 创建一个定时器，用于定期检查低头
        self.timer2 = QtCore.QTimer()
        self.timer2.timeout.connect(self.monitor_pose)
        self.timer2.start(1000)  # 每 1000 毫秒更新一次


    def monitor_pose(self):
        if self.saved_frame_ola < -15.0:
            self.infoBrowser.append("当前俯仰角为：{:.2f}\n危险！危险！".format(float(self.saved_frame_ola)))
            self.player.play()
        else:
            self.player.stop()

    def update_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("无法读取摄像头帧")
            return

        dets = face_detection(frame,self.net,self.device)

        # show image
        for b in dets:
            vis_thres = 0.6
            if b[4] < vis_thres:
                continue

            frame, p_deg = get_pose(frame,self.model,b,self.device)

            self.saved_frame_ola = p_deg

        # 将 OpenCV 图像转换为 QImage
        height, width, channel = frame.shape
        print(frame.shape)
        bytesPerLine = 3 * width
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        qImg = QtGui.QImage(cv_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        # 将 QImage 转换为 QPixmap 并显示
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.scene.clear()  # 清除之前的图像
        self.scene.addPixmap(pixmap)  # 添加新图像

    def clear_image(self):
        # 清除图像
        self.timer.stop()
        self.scene.clear()
        #self.infoBrowser.append("清除图像")

    def closeEvent(self, event):
        # 关闭窗口时释放摄像头资源
        if hasattr(self, 'cam'):
            self.cam.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MyWindow()
    ui.setupUi(MainWindow)
    ui.initial()
    MainWindow.show()
    sys.exit(app.exec_())