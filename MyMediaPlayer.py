import sys
import vlc
import os
import numpy as np
import threading
import time

APPTYPE=1
FILE = r"C:\Users\h01928\Documents\GIT_codes\MyMediaPlayer\media\video2.mp4"
current_running_speed = -1

# https://github.com/oaubert/python-vlc

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

            self.hbuttonbox = QHBoxLayout()
            self.hbuttonbox.addStretch(1)
            self.vbuttonbox = QVBoxLayout()
            self.vbuttonbox.addStretch(1)

            self.playbutton = QPushButton("Play",self)
            self.hbuttonbox.addWidget(self.playbutton)
            self.playbutton.clicked.connect(self.play_pause)

            self.stopbutton = QPushButton("Stop",self)
            self.hbuttonbox.addWidget(self.stopbutton)
            self.stopbutton.clicked.connect(self.stop)

            self.positionslider = QSlider(Qt.Horizontal, self)
            self.positionslider.setToolTip("Position")
            self.positionslider.setMaximum(1000)
            self.positionslider.setMinimumWidth(500)
            self.positionslider.setTickInterval(1)
            self.positionslider.sliderMoved.connect(self.set_position)
            self.positionslider.sliderPressed.connect(self.set_position)

            self.timer = QTimer(self)
            self.timer.setInterval(200)
            self.timer.timeout.connect(self.update_ui)

            self.timetxt = QLabel('0', self)
            self.timetxt.setFont(QFont('Times',12))
            self.timetxt.setAlignment(Qt.AlignCenter)
            self.hbuttonbox.addWidget(self.timetxt)

            self.speedtxt = QLabel('0', self)
            self.speedtxt.setFont(QFont('Times',14))
            self.speedtxt.setAlignment(Qt.AlignCenter)
            self.hbuttonbox.addWidget(self.speedtxt)

            self.vbuttonbox.addLayout(self.hbuttonbox)
            self.setLayout(self.vbuttonbox)

            menu_bar = self.menuBar()
            # File menu
            file_menu = menu_bar.addMenu("File")
            # Add actions to file menu
            open_action = QAction("Open video", self)
            #close_action = QAction("Close App", self)
            file_menu.addAction(open_action)
            #file_menu.addAction(close_action)
            open_action.triggered.connect(self.open_file)
            #close_action.triggered.connect(sys.exit)

            # media object
            media = vlc.Media(FILE)
            # setting media to the media player
            self.videoPlayer.set_media(media)
            #self.videoPlayer.play()
            self.mainFrame.setLayout(t_lay_parent)

            self.default_speed = 10.0  # km/h
            self.is_paused = True
            self.update_speed()
            self.playrate = 1.0

            self.show()

        def resizeEvent(self, event):
            x = self.rect().getCoords()
            self.playbutton.move(10, x[3] - 35)
            self.stopbutton.move(120, x[3] - 35)
            self.positionslider.move(250, x[3] - 35)
            self.timetxt.move(x[2]-250, x[3] - 35)
            self.timetxt.setFixedWidth(150)
            self.speedtxt.move(x[2] - 110, x[3] - 35)
            self.positionslider.setFixedWidth(x[2]-500)

            QMainWindow.resizeEvent(self, event)

        def set_position(self):
            """Set the movie position according to the position slider.
            """

            # The vlc MediaPlayer needs a float value between 0 and 1, Qt uses
            # integer variables, so you need a factor; the higher the factor, the
            # more precise are the results (1000 should suffice).

            # Set the media position to where the slider was dragged
            self.timer.stop()
            pos = self.positionslider.value()
            self.videoPlayer.set_position(pos / 1000.0)
            self.timer.start()

        def update_speed(self):
            self.speedtxt.setText(str('%1.2f' % current_running_speed))
            running_speed = min(current_running_speed,25) # capped
            if running_speed<0:
                self.speedtxt.setStyleSheet("color: red;  background-color: black")
            else:
                self.speedtxt.setStyleSheet("color: black;  background-color: white")
                self.playrate = max(0.001,running_speed/self.default_speed)

        def update_ui(self):
            """Updates the user interface"""

            # Set the slider's position to its corresponding media position
            # Note that the setValue function only takes values of type int,
            # so we must first convert the corresponding media position.
            media_pos = int(self.videoPlayer.get_position() * 1000)
            self.positionslider.setValue(media_pos)

            fps = -1
            try:
                fps = self.videoPlayer.get_fps()
            except:
                pass

            self.timetxt.setText(self.ms_to_timestr(max(0,self.videoPlayer.get_time())) + " (%1.0f)" % fps )
            self.update_speed()
            self.videoPlayer.set_rate(self.playrate)
            print("rate is %1.3f" % self.videoPlayer.get_rate())

            # No need to call this function if nothing is played
            if not self.videoPlayer.is_playing():
                self.timer.stop()

                # After the video finished, the play button stills shows "Pause",
                # which is not the desired behavior of a media player.
                # This fixes that "bug".
                if not self.is_paused:
                    self.stop()

        def ms_to_timestr(self,ms):
            sec = ms/1000.0
            hours = int(np.floor(sec/60/60))
            sec = sec - hours*60*60
            minutes = int(np.floor(sec/60))
            sec = sec - minutes*60
            return "%02i:%02i:%02.2f" % (hours,minutes,sec)

        def play_pause(self):
            """Toggle play/pause status
            """
            if self.videoPlayer.is_playing():
                self.videoPlayer.pause()
                self.playbutton.setText("Play")
                self.is_paused = True
                self.timer.stop()
            else:
                if self.videoPlayer.play() == -1:
                    self.open_file()
                    return

                self.videoPlayer.play()
                self.playbutton.setText("Pause")
                self.timer.start()
                self.is_paused = False

        def stop(self):
            """Stop player
            """
            self.videoPlayer.stop()
            self.playbutton.setText("Play")

        def open_file(self):
            """Open a media file in a MediaPlayer
            """
            dialog_txt = "Choose Media File"
            filename = QFileDialog.getOpenFileName(self, dialog_txt, os.path.expanduser('~'))
            if not filename:
                print("File was null")
                return

            # getOpenFileName returns a tuple, so use only the actual file name
            self.media = vlc.Media(filename[0])

            # Put the media in the media player
            self.videoPlayer.set_media(self.media)

            # Parse the metadata of the file
            self.media.parse()

            # Set the title of the track as window title
            self.setWindowTitle(self.media.get_meta(0))

            self.videoPlayer.audio_set_volume(0)
            self.playrate = 1.0

            self.play_pause()

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.LeftButton:
                if self.windowState() == Qt.WindowNoState:
                    #self.videoFrame1.hide()
                    self.videoFrame.show()
                    self.setWindowState(Qt.WindowFullScreen)
                else:
                    #self.videoFrame1.show()
                    self.setWindowState(Qt.WindowNoState)

# simulate treadmill speed input
def speed_function():
    global current_running_speed

    time.sleep(10)
    print("started speed tracking function!")
    running_speed = 0
    current_running_speed = running_speed
    t = 0
    step = 0.5
    default_running_speed = 10.0 # as km/h
    while True:

        running_speed = default_running_speed*(1+np.sin(t/30.0))
        running_speed += 0.05*(np.random.rand()*2 - 1.0)
        running_speed = np.maximum(0,running_speed)
        rate = running_speed/default_running_speed
        #print("time %f: Changed running speed to %f with rate %f" % (t, running_speed,rate))

        current_running_speed = running_speed
        time.sleep(step)
        t+=step
        if t>5000:
            current_running_speed = 0
            break

    print("tracking stopped!")

if __name__ == "__main__":

    # external thread that updates current_running_speed
    speed_thread = threading.Thread(target=speed_function, args=())
    speed_thread.start()

    if APPTYPE==0:
        app = QApplication(sys.argv)
        ex = App()
        sys.exit(app.exec_())
    else:
        app = QApplication(sys.argv)
        app.setApplicationName("Treadmill Player")
        window = MainWindow()
        app.exec_()


