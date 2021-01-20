import sys
import vlc

APPTYPE=1
FILE = r"C:\Users\h01928\Documents\GIT_codes\MyMediaPlayer\media\video2.mp4"

if APPTYPE==0: # simple test application

    import sys
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import pyqtSlot

    from PyQt5.QtWidgets import (QWidget, QPushButton,QHBoxLayout, QVBoxLayout, QApplication)

    class App(QWidget):

        def __init__(self):
            super().__init__()
            self.title = 'PyQt5 button - pythonspot.com'
            self.left = 100
            self.top = 100
            self.width = 320
            self.height = 200
            self.initUI()

        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)

            okButton = QPushButton("OK")
            cancelButton = QPushButton("Cancel")

            hbox = QHBoxLayout()
            hbox.addStretch(1)
            hbox.addWidget(okButton)
            hbox.addWidget(cancelButton)

            vbox = QVBoxLayout()
            vbox.addStretch(1)
            vbox.addLayout(hbox)

            self.setLayout(vbox)

            # button = QPushButton('PyQt5 button', self)
            # button.setToolTip('This is an example button')
            # button.move(10,170)
            # button.clicked.connect(self.on_click)

            self.show()

        @pyqtSlot()
        def on_click(self):
            print('PyQt5 button click')

else: # the actual application

    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtWebEngineWidgets import *
    from PyQt5.QtPrintSupport import *

    class MainWindow(QMainWindow):
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.sizeHint = lambda: QSize(1280, 900)
            self.setMinimumSize(QSize(300, 200))

            self.move(100, 10)
            self.mainFrame = QFrame()
            self.setCentralWidget(self.mainFrame)
            t_lay_parent = QHBoxLayout()
            t_lay_parent.setContentsMargins(0,0,0,50)

            if 1:
                self.videoFrame = QFrame()
                self.videoFrame.mouseDoubleClickEvent = self.mouseDoubleClickEvent
                t_lay_parent.addWidget(self.videoFrame)
                self.vlcInstance = vlc.Instance(['--video-on-top'])
                self.videoPlayer = self.vlcInstance.media_player_new()
                self.videoPlayer.video_set_mouse_input(False)
                self.videoPlayer.video_set_key_input(False)
                self.videoPlayer.set_mrl("http://xxx.xxx.xxx.xxx", "network-caching=300")
                self.videoPlayer.audio_set_mute(True)
                if sys.platform.startswith('linux'):  # for Linux using the X Server
                    self.videoPlayer.set_xwindow(self.videoFrame.winId())
                elif sys.platform == "win32":  # for Windows
                    self.videoPlayer.set_hwnd(self.videoFrame.winId())
                elif sys.platform == "darwin":  # for MacOS
                    self.videoPlayer.set_nsobject(int(self.videoFrame.winId()))

            self.hbuttonbox = QHBoxLayout()
            self.hbuttonbox.addStretch(1)
            self.vbuttonbox = QVBoxLayout()
            self.vbuttonbox.addStretch(1)

            self.playbutton = QPushButton("Play",self)
            x = self.rect().getCoords()
            self.hbuttonbox.addWidget(self.playbutton)
            self.playbutton.clicked.connect(self.play_pause)

            self.stopbutton = QPushButton("Stop",self)
            self.hbuttonbox.addWidget(self.stopbutton)
            self.stopbutton.clicked.connect(self.stop)

            self.vbuttonbox.addLayout(self.hbuttonbox)
            self.setLayout(self.vbuttonbox)

            # media object

            media = vlc.Media(FILE)
            # setting media to the media player
            self.videoPlayer.set_media(media)
            #self.videoPlayer.play()
            self.mainFrame.setLayout(t_lay_parent)

            self.show()

        def resizeEvent(self, event):
            x = self.rect().getCoords()
            self.playbutton.move(10, x[3] - 35)
            self.stopbutton.move(120, x[3] - 35)
            QMainWindow.resizeEvent(self, event)

        def play_pause(self):
            """Toggle play/pause status
            """
            if self.videoPlayer.is_playing():
                self.videoPlayer.pause()
                self.playbutton.setText("Play")
                self.is_paused = True
                #self.timer.stop()
            else:
                if self.videoPlayer.play() == -1:
                    self.open_file()
                    return

                self.videoPlayer.play()
                self.playbutton.setText("Pause")
                #self.timer.start()
                self.is_paused = False

        def stop(self):
            """Stop player
            """
            self.videoPlayer.stop()
            self.playbutton.setText("Play")

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.LeftButton:
                if self.windowState() == Qt.WindowNoState:
                    #self.videoFrame1.hide()
                    self.videoFrame.show()
                    self.setWindowState(Qt.WindowFullScreen)
                else:
                    #self.videoFrame1.show()
                    self.setWindowState(Qt.WindowNoState)

if __name__ == "__main__":

    if APPTYPE==0:
        app = QApplication(sys.argv)
        ex = App()
        sys.exit(app.exec_())

    else:

        app = QApplication(sys.argv)
        app.setApplicationName("Treadmill Player")
        window = MainWindow()
        app.exec_()

    # else:
    #     # importing vlc module
    #     import vlc
    #     import numpy as np
    #     # importing time module
    #     import time
    #
    #     # creating vlc media player object
    #     media_player = vlc.MediaPlayer()
    #
    #     # media object
    #     FILE = r"C:\Users\h01928\Documents\GIT_codes\MyMediaPlayer\media\video2.mp4"
    #     media = vlc.Media(FILE)
    #
    #     # setting media to the media player
    #     media_player.set_media(media)
    #
    #     # start playing video
    #     media_player.play()
    #
    #     #media_player.toggle_fullscreen()
    #     # wait so the video can be played for 5 seconds
    #     # irrespective for length of video
    #     t = 0
    #     step = 1.0
    #     running_speed = 8.0 # as km/h
    #     conversion_rate = 1.0/8.0
    #     while True:
    #         running_speed = 10.0 + 8*np.sin(t/5.0)
    #         running_speed += 0.25*(np.random.rand()*2 - 1.0)
    #         running_speed = np.maximum(0,running_speed)
    #
    #         rate = running_speed*conversion_rate
    #         media_player.set_rate(rate)
    #
    #         print("time %f with running speed %f and rate %f" % (t, running_speed,rate))
    #
    #         time.sleep(step)
    #         t+=step
    #         if t>60:
    #             break

