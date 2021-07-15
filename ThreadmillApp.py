import cv2
import glob
import os
import numpy as np
import threading
import time
import sys
import pickle
import configparser
import matplotlib.pyplot as plt

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

config_data = configparser.ConfigParser()
CONFIG_FILE = "config_params.cfg"
config_data.read(CONFIG_FILE)

os.add_dll_directory(config_data.get("paths","vlc_path"))
import vlc

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.models import load_model

# global parameters
DIGIT_MODEL = config_data.get("paths","model_path") # path to saved Keras model
PREDICT_INTERVAL = float(config_data.get("main","predict_interval")) # in ms
DEFAULT_PATH = config_data.get("paths","video_path") # path to running videos (MP4)
SMOOTH_WINDOW = float(config_data.get("main","smooth_window")) # path to running videos (MP4)
DEFAULT_SPEED = float(config_data.get("main","default_speed"))
MAX_SPEED = float(config_data.get("main","max_speed"))

running_speed_history = [] # keep track of previous speeds
current_running_speed = -1 # dynamic speed to change video rate

# FOR DEBUGGING AND DEVELOPMENT
WEBCAM_VIDEO = "" #r'D:\JanneK\Documents\git_repos\MyMediaPlayer-main\mydata\WIN_20210713_20_33_27_Pro.mp4'
LOAD_MODEL = True # set false for faster loading in debugging

# this is the window that playes the selected video with dynamic playrate
class VideoWindow(QMainWindow):

    def __init__(self,initial_video = None,default_speed = 6.0,*args, **kwargs):
        super(VideoWindow, self).__init__(*args, **kwargs)

        self.default_speed = default_speed  # km/h
        self.sizeHint = lambda: QSize(1280, 900)
        self.setMinimumSize(QSize(300, 200))

        self.move(100, 10)
        self.mainFrame = QFrame()
        self.setCentralWidget(self.mainFrame)
        t_lay_parent = QHBoxLayout()
        t_lay_parent.setContentsMargins(0,0,0,40)

        self.videoFrame = QFrame()
        self.videoFrame.mouseDoubleClickEvent = self.mouseDoubleClickEvent
        t_lay_parent.addWidget(self.videoFrame)
        self.vlcInstance = vlc.Instance('--video-on-top','--disable-screensaver','--high-priority','--loop') # could put in instance parameters here
        self.videoPlayer = self.vlcInstance.media_player_new()
        self.videoPlayer.video_set_mouse_input(False)
        self.videoPlayer.video_set_key_input(False)
        self.videoPlayer.set_mrl("http://xxx.xxx.xxx.xxx", "network-caching=300")
        #self.videoPlayer.audio_set_mute(True)
        self.videoPlayer.audio_set_volume(30)
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
        self.timer.setInterval(250) # polling every 250ms
        self.media_duration=-1
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

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
        '''
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
        '''
        # media object
        self.media = None
        self.media_duration=-1
        if initial_video is not None and os.path.exists(initial_video):
            print('Loaded default media')
            self.media = vlc.Media(initial_video)
            # setting media to the media player
            self.videoPlayer.set_media(self.media)
            #self.vlcInstance.vlm_set_loop(initial_video, True) # didn't work!
            self.videoPlayer.play()

        #self.videoPlayer.playButton()
        self.mainFrame.setLayout(t_lay_parent)

        self.min_playrate = float(config_data["main"]["min_playrate"])
        self.update_speed()
        self.playrate = 1.0

        self.show()

    def closeEvent(self, event):
        pass

    def resizeEvent(self, event):
        x = self.rect().getCoords()
        self.playbutton.move(10, x[3] - 35)
        self.stopbutton.move(120, x[3] - 35)
        self.positionslider.move(250, x[3] - 35)
        self.timetxt.move(x[2]-300, x[3] - 35)
        self.timetxt.setFixedWidth(150)
        self.speedtxt.move(x[2] - 160, x[3] - 35)
        self.speedtxt.setFixedWidth(150)
        self.positionslider.setFixedWidth(x[2]-560)
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
        global current_running_speed
        if current_running_speed<0:
            self.speedtxt.setStyleSheet("color: red;  background-color: black")
        else:
            self.speedtxt.setStyleSheet("color: black;  background-color: white")
            self.playrate = max(self.min_playrate,current_running_speed/self.default_speed)
        self.speedtxt.setText('%1.2fkm/h (%.1f)' % (current_running_speed,self.playrate))

    def update_ui(self):
        """Updates the user interface"""

        # Set the slider's position to its corresponding media position
        # Note that the setValue function only takes values of type int,
        # so we must first convert the corresponding media position.
        #print('executed update_ui')
        media_pos = int(self.videoPlayer.get_position() * 1000)
        self.positionslider.setValue(media_pos)

        if self.media_duration<=5:
            self.media_duration = self.videoPlayer.get_length()/1000

        if self.media_duration > 10 and abs(media_pos-self.media_duration)<5:
            self.videoPlayer.set_position(0)

        fps = -1
        try:
            fps = self.videoPlayer.get_fps()
        except:
            pass

        fps = fps if fps <200 and fps>10 else -1
        self.timetxt.setText(self.ms_to_timestr(max(0,self.videoPlayer.get_time())) + " (%1.0f)" % fps )
        self.update_speed()
        self.videoPlayer.set_rate(self.playrate)
        #print("rate is %1.3f" % self.videoPlayer.get_rate())

        # No need to call this function if nothing is played
        if not self.videoPlayer.is_playing():
            self.timer.stop()
            '''
            # After the video finished, the play button stills shows "Pause",
            # which is not the desired behavior of a media player.
            # This fixes that "bug".
            if not self.is_paused:
                self.stop()
            '''

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
            self.timer.stop()
        else:
            if self.videoPlayer.playButton() == -1:
                self.open_file()
            else:
                self.videoPlayer.play()
                self.playbutton.setText("Pause")
                self.timer.start()

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
        print("file '%s' loaded" % filename[0],flush=True)
        # getOpenFileName returns a tuple, so use only the actual file name
        self.media = vlc.Media(filename[0])

        # Put the media in the media player
        self.videoPlayer.set_media(self.media)
        #self.vlcInstance.vlm_set_loop(self.media, True)

        # Parse the metadata of the file
        self.media.parse()

        # Set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))

        self.videoPlayer.audio_set_volume(0)
        self.playrate = 1.0
        self.media_duration = -1
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

# this is the main window that contains list of videos and webcam stream
class MainWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):

        #myGUI = QtWidgets.QDialog()
        myGUI = self
        myGUI.setObjectName("myGUI")
        myGUI.resize(875, 461)
        myGUI.setMaximumWidth(myGUI.width())
        myGUI.setMaximumHeight(myGUI.height())

        self.precision_group = QtWidgets.QButtonGroup(myGUI)
        self.precision0 = QtWidgets.QRadioButton(myGUI)
        self.precision0.setGeometry(QtCore.QRect(800, 280, 41, 31))
        self.precision0.setObjectName("precision0")
        self.precision1 = QtWidgets.QRadioButton(myGUI)
        self.precision1.setGeometry(QtCore.QRect(800, 310, 41, 17))
        self.precision1.setObjectName("precision1")
        self.precision2 = QtWidgets.QRadioButton(myGUI)
        self.precision2.setGeometry(QtCore.QRect(800, 330, 41, 17))
        self.precision2.setObjectName("precision2")
        self.precision3 = QtWidgets.QRadioButton(myGUI)
        self.precision3.setGeometry(QtCore.QRect(800, 350, 41, 17))
        self.precision3.setObjectName("precision3")
        self.precision4 = QtWidgets.QRadioButton(myGUI)
        self.precision4.setGeometry(QtCore.QRect(800, 370, 41, 16))
        self.precision4.setObjectName("precision4")

        self.precision1.setChecked(1)
        self.precision_group.addButton(self.precision0,0)
        self.precision_group.addButton(self.precision1,1)
        self.precision_group.addButton(self.precision2,2)
        self.precision_group.addButton(self.precision3,3)
        self.precision_group.addButton(self.precision4,4)
        self.precision_group.buttonClicked.connect(self.precision_event)

        self.digitcount_group = QtWidgets.QButtonGroup(myGUI)
        self.digitcount2 = QtWidgets.QRadioButton(myGUI)
        self.digitcount2.setGeometry(QtCore.QRect(800, 170, 41, 17))
        self.digitcount2.setObjectName("digitcount2")
        self.digitcount4 = QtWidgets.QRadioButton(myGUI)
        self.digitcount4.setGeometry(QtCore.QRect(800, 210, 41, 17))
        self.digitcount4.setObjectName("digitcount4")
        self.digitcount5 = QtWidgets.QRadioButton(myGUI)
        self.digitcount5.setGeometry(QtCore.QRect(800, 230, 41, 16))
        self.digitcount5.setObjectName("digitcount5")
        self.digitcount1 = QtWidgets.QRadioButton(myGUI)
        self.digitcount1.setGeometry(QtCore.QRect(800, 140, 41, 31))
        self.digitcount1.setObjectName("digitcount1")
        self.digitcount3 = QtWidgets.QRadioButton(myGUI)
        self.digitcount3.setGeometry(QtCore.QRect(800, 190, 41, 17))
        self.digitcount3.setObjectName("digitcount3")
        self.digitcount_group.buttonClicked.connect(self.digitcount_event)

        self.digitcount3.setChecked(1)
        self.digitcount_group.addButton(self.digitcount1,1)
        self.digitcount_group.addButton(self.digitcount2,2)
        self.digitcount_group.addButton(self.digitcount3,3)
        self.digitcount_group.addButton(self.digitcount4,4)
        self.digitcount_group.addButton(self.digitcount5,5)

        self.label = QtWidgets.QLabel(myGUI)
        self.label.setGeometry(QtCore.QRect(780, 120, 51, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(myGUI)
        self.label_2.setGeometry(QtCore.QRect(790, 260, 51, 20))
        self.label_2.setObjectName("label_2")

        self.regionselection = QtWidgets.QLabel(myGUI)
        self.regionselection.setGeometry(QtCore.QRect(500,70,150, 20))
        self.regionselection.setObjectName("regionselection")
        self.regionselection.setText("No ROI selected")

        self.lcdNumber = QtWidgets.QLCDNumber(myGUI)
        self.lcdNumber.setGeometry(QtCore.QRect(630, 15, 131, 31))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setStyleSheet("QLCDNumber { background-color: rgb(255, 255,255);color: rgb(0,255,0);font-weight: bold;}")

        self.frames_per_sec = QtWidgets.QLabel(myGUI)
        self.frames_per_sec.setGeometry(QtCore.QRect(660, 60, 100, 28))
        self.frames_per_sec.setText("0 frames/sec")

        self.medianbox = QtWidgets.QCheckBox(myGUI)
        self.medianbox.setGeometry(QtCore.QRect(780, 400, 81, 17))
        self.medianbox.setTristate(False)
        self.medianbox.setChecked(False)
        self.medianbox.setObjectName("medianbox")

        self.tracking = QtWidgets.QCheckBox(myGUI)
        self.tracking.setGeometry(QtCore.QRect(780,25, 81, 17))
        self.tracking.setTristate(False)
        self.tracking.setChecked(True)
        self.tracking.setObjectName("tracking")
        self.tracking.clicked.connect(self.start_camera)

        self.boxdetect = QtWidgets.QCheckBox(myGUI)
        self.boxdetect.setGeometry(QtCore.QRect(780, 420, 81, 17))
        self.boxdetect.setTristate(False)
        self.boxdetect.setChecked(True)
        self.boxdetect.setObjectName("boxdetect")
        self.graphicsView = QtWidgets.QLabel(myGUI)
        self.graphicsView.setGeometry(QtCore.QRect(350, 90, 411, 331))
        self.graphicsView.setObjectName("graphicsView")
        self.videopathEdit = QtWidgets.QTextEdit(myGUI)
        self.videopathEdit.setGeometry(QtCore.QRect(90, 10, 421, 31))
        self.videopathEdit.setObjectName("textEdit")
        self.videopathEdit.setText(DEFAULT_PATH)
        self.videopathEdit.textChanged.connect(self.updatefiles)

        self.label_3 = QtWidgets.QLabel(myGUI)
        self.label_3.setGeometry(QtCore.QRect(550,20, 71, 16))
        self.label_3.setObjectName("label_3")
        self.listWidget = QtWidgets.QListWidget(myGUI)
        self.listWidget.setGeometry(QtCore.QRect(20, 90, 311, 341))
        self.listWidget.setObjectName("listWidget")
        self.listWidget.itemClicked.connect(self.onClickedFile)

        self.source_group = QtWidgets.QButtonGroup(myGUI)
        self.source0 = QtWidgets.QRadioButton(myGUI)
        self.source0.setGeometry(QtCore.QRect(780, 50, 71, 17))
        self.source0.setObjectName("source0")
        self.source1 = QtWidgets.QRadioButton(myGUI)
        self.source1.setGeometry(QtCore.QRect(780, 70, 71, 17))
        self.source1.setObjectName("source1")
        self.source_group.addButton(self.source0,0)
        self.source_group.addButton(self.source1,1)
        self.source_group.buttonClicked.connect(self.source_event)

        self.source0.setChecked(1)

        self.label_4 = QtWidgets.QLabel(myGUI)
        self.label_4.setGeometry(QtCore.QRect(20, 10, 71, 16))
        self.label_4.setObjectName("label_4")
        self.playButton = QtWidgets.QPushButton(myGUI)
        self.playButton.setGeometry(QtCore.QRect(410, 60, 75, 23))
        self.playButton.setObjectName("play")
        self.playButton.clicked.connect(self.play_movie)
        self.playButton.setEnabled(False)

        self.videospeedEdit = QtWidgets.QTextEdit(myGUI)
        self.videospeedEdit.setGeometry(QtCore.QRect(340,60,65, 23))
        self.videospeedEdit.setObjectName("videospeed")
        self.videospeedEdit.setText("")
        self.videospeedEdit.textChanged.connect(self.edited_speed)
        self.videospeedEdit.setEnabled(False)

        self.currentvideoEdit = QtWidgets.QPlainTextEdit(myGUI)
        self.currentvideoEdit.setGeometry(QtCore.QRect(20, 60, 311,21))
        self.currentvideoEdit.setObjectName("plainTextEdit")
        self.currentvideoEdit.setReadOnly(True)

        self.mouse_pos = None

        try:
            x1,y1,x2,y2 = [int(x) for x in config_data.get("main", "roi").split(",")]
            self.selected_ROI = True
            self.coordinate1 = (x1,y1)
            self.coordinate2 = (x2,y2)
        except:
            self.selected_ROI = False
            self.coordinate1 = None
            self.coordinate2 = None

        self.frame_shape = (self.graphicsView.width(), self.graphicsView.height())
        self.frame_scaling = None

        self.retranslateUi(myGUI)
        QtCore.QMetaObject.connectSlotsByName(myGUI)

        self.updatefiles()
        self.chosen_file=None

        self.camera_thread = None
        myGUI.setMouseTracking(True)

        self.digitcount=3
        self.set_running_speed(current_running_speed)
        self.precision=1
        self.videowindow=None
        self.starttime = time.time()
        self.defaultspeed=DEFAULT_SPEED
        self.start_camera()
        self.show()

    def edited_speed(self):
        self.defaultspeed = float(self.videospeedEdit.toPlainText())

    def set_running_speed(self,val):
        self.lcdNumber.setNumDigits(self.digitcount+1)
        self.lcdNumber.display(val)

    def digitcount_event(self,e):
        self.digitcount = self.digitcount_group.id(e)
        print("changed digitcount to %i" % self.digitcount)

    def source_event(self,e):
        print("changed source to %i" % self.source_group.checkedId())

    def precision_event(self,e):
        self.precision = self.precision_group.id(e)
        print("changed precision to %i" % self.precision)

    def mousemove(self,e):
        x = e.x()
        y = e.y()
        if self.frame_shape is not None:
            x = x - self.frame_padding[0]
            y = y - self.frame_padding[1]
            if x > 0 and y > 0 and x <= self.frame_shape[0] and y <= self.frame_shape[1]:
                self.mouse_pos = (x, y)

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

    # webcam frame click release, gets coordinate 2
    def mouseup(self,e):
        # Record ending (x,y) coordintes on left mouse bottom release
        if e.button()==1:
            if self.frame_shape is not None:
                x = e.x() - self.frame_padding[0]
                y = e.y() - self.frame_padding[1]
                self.coordinate2 = (x,y)
                print("mouse coord 2 (x, y) = (%i, %i)" % (self.coordinate2[0], self.coordinate2[1]))
                self.selected_ROI = True

                # update config file
                config_data["main"]["roi"] = "%i,%i,%i,%i" % (self.coordinate1[0],self.coordinate1[1],self.coordinate2[0],self.coordinate2[1])
                with open(CONFIG_FILE, 'w') as configfile:
                    config_data.write(configfile)

    # clicked webcam frame
    def mousedown(self, e):
        # Record starting (x,y) coordinates on left mouse button click
        if e.button()==1:
            if self.frame_shape is not None:
                x = e.x() - self.frame_padding[0]
                y = e.y() - self.frame_padding[1]
                self.coordinate1 = (x, y)
                self.coordinate2 = None
                self.selected_ROI = False
                print("mouse coord 1 (x, y) = (%i, %i)" % (self.coordinate1[0],self.coordinate1[1]))
        # Clear drawing boxes on right mouse button click
        elif e.button()==2:
            self.coordinate1 = None
            self.coordinate2 = None
            self.selected_ROI = False
            self.regionselection.setText("No ROI selected")

    # begin recording
    def start_camera(self):
        global current_running_speed
        if self.tracking.isChecked():
            device = self.source_group.checkedId()
            self.camera_thread = threading.Thread(target=self.camera_update, args=[device])
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            current_running_speed=self.defaultspeed
            self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
            self.frames_per_sec.setText("-")

    # function to capture and analyze webcam stream (running as separate thread)
    def camera_update(self,device):
        global running_speed_history,current_running_speed
        if len(WEBCAM_VIDEO):
            print("!! Loading webcam video instead of actual stream (DEBUGGING)!!")
            self.capture = cv2.VideoCapture(WEBCAM_VIDEO)
        else:
            self.capture = cv2.VideoCapture(device,cv2.CAP_DSHOW)
        old_device = device
        rectangles = None
        last_prediction_time = time.time() - PREDICT_INTERVAL
        print("Starting video capture")
        while True:
            if self.capture.isOpened():
                if self.source_group.checkedId() != old_device or not(self.tracking.isChecked()):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    current_running_speed = self.defaultspeed
                    self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
                    return
                # Read frame

                (status, frame) = self.capture.read()
                self.frame_scaling = (frame.shape[1] / self.frame_shape[0], frame.shape[0] / self.frame_shape[1])

                now = time.time()
                if self.selected_ROI:

                    x1 = min(self.coordinate1[0],self.coordinate2[0])
                    x2 = max(self.coordinate1[0], self.coordinate2[0])
                    y1 = min(self.coordinate1[1],self.coordinate2[1])
                    y2 = max(self.coordinate1[1], self.coordinate2[1])
                    # scale coordinated to actual frame
                    ROI = [x1*self.frame_scaling[0],y1*self.frame_scaling[1],x2*self.frame_scaling[0],y2*self.frame_scaling[1],self.digitcount] # [x1,y1,x2,y2,digit_count]
                    ROI = [int(x) for x in ROI]

                    # if enough time passed, analyze digits
                    if (now - last_prediction_time)*1000 > PREDICT_INTERVAL:
                        predicted_digits,rectangles = predict_digit(frame, ROI,self.medianbox.isChecked(),self.boxdetect.isChecked())
                        if predicted_digits is None:
                            # failed, just use last valid prediction
                            if len(running_speed_history)>0:
                                predicted_speed = running_speed_history[-1][1]
                            else:
                                predicted_speed = current_running_speed
                        else:
                            # convert individual digits to float using selected precision
                            predicted_digits_str = "".join([str(x) if x>-1 else "0" for x in predicted_digits])
                            predicted_digits_str = predicted_digits_str[0:(len(predicted_digits)-self.precision)] + "." + predicted_digits_str[-self.precision:]
                            predicted_speed = float(predicted_digits_str)

                        timestamp = now - self.starttime
                        running_speed_history.append((timestamp, predicted_speed))
                        # a function to obtain speed, could include some fancy smoothing and error-checking
                        current_running_speed = speed_estimator(timestamp,running_speed_history)

                        # only keep last 100 measurements
                        if len(running_speed_history)>100:
                            running_speed_history = running_speed_history[-100:]

                        self.lcdNumber.display("{1:,.{0}f}".format(self.precision,current_running_speed))
                        self.frames_per_sec.setText("%.2f frames/sec" % (1.0 / (max(0.0001,time.time()-now))))
                        last_prediction_time = now

                    frame = cv2.rectangle(frame, (ROI[0],ROI[1]),(ROI[2],ROI[3]), (0, 255, 0),4)
                    if rectangles is not None:
                        for rect in rectangles:
                            frame = cv2.rectangle(frame,
                                                (rect[0],rect[2]),
                                                (rect[1],rect[3]),
                                                (0, 200,200),3)

                    self.regionselection.setText("ROI (%i,%i,%i,%i)" % (x1,y1,x2,y2))

                    #for r in self.digit_ROIs:
                    #    self.frame = cv2.rectangle(self.frame, (r[1][0]+x1,r[1][1]+y1), (r[1][2]+x1,r[1][3]+y1), (0,50,255), 2)
                elif self.coordinate1 is not None:
                    x1 = int(min(self.coordinate1[0],self.mouse_pos[0])*self.frame_scaling[0])
                    x2 = int(max(self.coordinate1[0], self.mouse_pos[0])*self.frame_scaling[0])
                    y1 = int(min(self.coordinate1[1],self.mouse_pos[1])*self.frame_scaling[1])
                    y2 = int(max(self.coordinate1[1], self.mouse_pos[1])*self.frame_scaling[1])
                    rectangles = None
                    frame = cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2)

                self.image = frame
                self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                pixmap = QtGui.QPixmap.fromImage(self.image)
                pixmap = pixmap.scaled(self.graphicsView.width(), self.graphicsView.height(), QtCore.Qt.KeepAspectRatio)
                self.graphicsView.setPixmap(pixmap)
                self.graphicsView.setMouseTracking(True)
                self.frame_padding = ((self.graphicsView.width() - self.frame_shape[0]) / 2, (self.graphicsView.height() - self.frame_shape[1]) / 2)
                self.graphicsView.mouseMoveEvent = self.mousemove
                self.graphicsView.mousePressEvent = self.mousedown
                self.graphicsView.mouseReleaseEvent = self.mouseup

                key = cv2.waitKey(1)

            else:
                print("Capture terminated!")
                self.capture.release()
                cv2.destroyAllWindows()
                current_running_speed = self.defaultspeed
                self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))

    def updatefiles(self):
        files = glob.glob(self.videopathEdit.toPlainText() + os.sep + "*.mp4")
        files = [f.replace(self.videopathEdit.toPlainText() + os.sep, '') for f in files]
        for f in files:
            self.listWidget.addItem(f)
        files_with_rates = [[f,str(DEFAULT_SPEED)] for f in files]

        old_files = {k:config_data["files"].get(k).split(":") for k in list(config_data["files"])}
        for k1,old_file in old_files.items():
            for k2,file_with_rate in enumerate(files_with_rates):
                if old_file[0]==file_with_rate[0]:
                    files_with_rates[k2][1] = old_files[k1][1]
        config_data["files"] = {i:":".join(f) for i,f in enumerate(files_with_rates)}
        with open(CONFIG_FILE, 'w') as configfile:
            config_data.write(configfile)

    def onClickedFile(self,item):
        self.chosen_file = [self.videopathEdit.toPlainText(),item.text()]
        self.currentvideoEdit.setPlainText(item.text())
        self.playButton.setEnabled(True)
        self.videospeedEdit.setEnabled(True)
        saved_files = {k: config_data["files"].get(k).split(":") for k in list(config_data["files"])}
        for k,file in saved_files.items():
            if self.chosen_file[1] in file[0]:
                self.defaultspeed = float(file[1])
        self.videospeedEdit.setText(str(self.defaultspeed))

    def retranslateUi(self, myGUI):
        _translate = QtCore.QCoreApplication.translate
        myGUI.setWindowTitle(_translate("myGUI", "TreadmillApp"))
        self.precision0.setText(_translate("myGUI", "0"))
        self.precision1.setText(_translate("myGUI", "1"))
        self.precision2.setText(_translate("myGUI", "2"))
        self.precision3.setText(_translate("myGUI", "3"))
        self.precision4.setText(_translate("myGUI", "4"))
        self.digitcount2.setText(_translate("myGUI", "2"))
        self.digitcount4.setText(_translate("myGUI", "4"))
        self.digitcount5.setText(_translate("myGUI", "5"))
        self.digitcount1.setText(_translate("myGUI", "1"))
        self.digitcount3.setText(_translate("myGUI", "3"))
        self.label.setText(_translate("myGUI", "digit count"))
        self.label_2.setText(_translate("myGUI", "precision"))
        self.medianbox.setText(_translate("myGUI", "median box"))
        self.tracking.setText(_translate("myGUI", "tracking"))
        self.boxdetect.setText(_translate("myGUI", "box detect"))
        self.label_3.setText(_translate("myGUI", "current speed"))
        self.source0.setText(_translate("myGUI", "source 0"))
        self.source1.setText(_translate("myGUI", "source 1"))
        self.label_4.setText(_translate("myGUI", "current path"))
        self.playButton.setText(_translate("myGUI", "Play selected"))

    def play_movie(self):
        saved_files = {k: config_data["files"].get(k).split(":") for k in list(config_data["files"])}
        for k,file in saved_files.items():
            if self.chosen_file[1] in file[0]:
                config_data["files"][str(k)] = ":".join([file[0],str(self.defaultspeed)])
        with open(CONFIG_FILE, 'w') as configfile:
            config_data.write(configfile)
        self.videowindow = VideoWindow((os.sep).join(self.chosen_file),self.defaultspeed)

    def closeEvent(self, event):
        if self.videowindow is not None:
            self.videowindow.close()

def speed_estimator(current_time,running_speed_history):
    prev_speeds = []
    for x in reversed(running_speed_history):
        if (current_time - x[0])<SMOOTH_WINDOW:
            prev_speeds.append(x[1])
        else:
            break
    display_speed = float(np.median(prev_speeds))
    display_speed = max(0.001, min(MAX_SPEED,display_speed))
    return display_speed

# given frame and ROI, return sub-images of digits
def image_preprocessor(img,ROI,use_prev_box,use_detection,prev_boxes=None):
    if prev_boxes is None:
        prev_boxes=[]
    ROI_w = ROI[2] - ROI[0]
    ROI_h = ROI[3] - ROI[1]
    img = img[ROI[1]:(ROI[3] + 1), ROI[0]:(ROI[2] + 1), :]
    img = np.flip(img, axis=0)
    img = np.flip(img, axis=1)
    # print("image size = ",str(img.shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # find contours
    if use_detection:
        contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  #
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [x for x in boxes if x[2] > img.shape[1] * 0.70 and x[3] > img.shape[0] * 0.70]
        boxes = [x for x in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)]
    add_to_hist = True
    if not(use_detection) or len(boxes) == 0:
        extra = 1
        boxes = [[extra,extra,ROI_w-2*extra,ROI_h-2*extra]]
        add_to_hist=False
    box = boxes[0]
    if add_to_hist:
        prev_boxes.append(box)
        if len(prev_boxes)>=51:
            prev_boxes=prev_boxes[-51:]
    if use_prev_box:
        box = [int(np.median([x[k] for x in prev_boxes])) for k in range(4)]
    img = img[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2]), :]
    dx = int(np.round(img.shape[1] / ROI[-1]))
    digits = []
    rectangles = []
    for part in range(ROI[-1]):
        rectangles.append([
            ROI[0] + ROI_w - (box[0] + (dx * (part + 1))),
            ROI[0]+ ROI_w - (box[0]+(dx * part)),
            ROI[1]+ROI_h-box[3],
            ROI[3]-box[1]])
        sub_img = img[:, (dx * part):(dx * (part + 1)), :]
        digits.append(sub_img)
    return digits,prev_boxes,rectangles

# resize sub-image with optional padding
def resize_image(img, size=(28,28),PAD = 0):
    # size = h,w
    interpolation = cv2.INTER_AREA
    size = size[0]-PAD,size[1]-PAD
    #extra_pad = int(PAD / 2)
    h, w = img.shape[:2]
    aspect_ratio = h/w
    new_aspect_ratio = size[0]/size[1]
    #c = img.shape[2] if len(img.shape)>2 else 1
    if h == w:
        new_im = cv2.resize(img, size, interpolation)
        #new_im = cv2.copyMakeBorder(mask,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        assert new_im.shape == IMG_SIZE
        return new_im
    if new_aspect_ratio<aspect_ratio: # new image is wider, so need padding to width
        tmp_size = [size[0],int(np.round(size[0]/aspect_ratio))]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[1]-new_im.shape[1]
        new_im = cv2.copyMakeBorder(new_im,0,0,int(P/2),0, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0, 0,0,size[1]-new_im.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right
    else:
        tmp_size = [int(np.round(size[1]/aspect_ratio)),size[1]]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[0]-new_im.shape[0]
        new_im = cv2.copyMakeBorder(new_im,int(P/2),0,0,0, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0,size[0]-new_im.shape[0],0,0, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, righ
    #new_im = cv2.copyMakeBorder(new_im,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right
    assert new_im.shape[:2] == IMG_SIZE,"resized image incorrect shape (%s)!" % str(new_im.shape)
    return new_im

# feed processed images to the predictor model
prev_boxes = []
predictor_model = None
def predict_digit(frame,ROI,use_prev_box,use_detection):
    global prev_boxes,predictor_model
    if predictor_model is None:
        return None,None
    frame_digits, prev_boxes, rectangles = image_preprocessor(frame,ROI,use_prev_box,use_detection,prev_boxes=prev_boxes)
    if len(frame_digits) != ROI[-1]:
        print("Failed to find individual digits (found %i digits our of required %i)!" % (len(frame_digits),ROI[-1]))
        return None,None
    frame_digits = [resize_image(x, IMG_SIZE) for x in frame_digits]
    frame_digits = np.stack(frame_digits, axis=0).astype(np.float32) / 255.0
    digits = predictor_model.predict(frame_digits)
    digits = np.argmax(digits, 1)
    digits[digits == 10] = -1
    return digits,rectangles

if __name__ == "__main__":
    if LOAD_MODEL:
        # load and test predictor model before starting the app
        print("Loading recognition model... ", end="")
        predictor_model = load_model(DIGIT_MODEL)
        IMG_SIZE = predictor_model.layers[1].input_shape[1:3]
        print("done")
        print("making dummy prediction to initialize predictor... ",end='')
        predictor_model.predict(np.random.rand(10,IMG_SIZE[0],IMG_SIZE[1],3))
        print("done")

    print("Starting application")
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    sys.exit(app.exec_())


