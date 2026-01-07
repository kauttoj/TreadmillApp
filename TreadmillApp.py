import cv2
import glob
import os
import numpy as np
import threading
import time
import datetime
from PIL import Image

import sys
import configparser

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from skimage.registration import phase_cross_correlation

config_data = configparser.ConfigParser(allow_no_value=True)
CONFIG_FILE = "config_params.cfg"
assert os.path.exists(CONFIG_FILE)>0,"Config file with name config_params.cfg not found in program path!"
config_data.read(CONFIG_FILE,encoding="UTF-8")

os.add_dll_directory(config_data.get("paths","vlc_path"))
import vlc

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # use CPU

if config_data.get("main","model_type") == 'tensorflow':
    from tensorflow.keras.models import load_model
elif config_data.get("main","model_type") == 'pytorch':
    import torch
    from pytorch_digits_model import Network
else:
    raise(AssertionError('Bad model type!'))

def set_value(entry,default):
    global config_data
    try:
        val = float(config_data.get("main",entry))
    except:
        val = default
    config_data["main"][entry] = str(val)
    return val

print('Loaded config file parameters are:')
for k in config_data.keys():
    if k!='files':
        print('Section = {0}'.format(k))
        for v in config_data[k].keys():
            print('... {0} = {1}'.format(v,config_data[k][v]))

# required, no defaults
DIGIT_MODEL = config_data.get("paths","model_path") # path to saved Keras model
DEFAULT_PATH = config_data.get("paths","video_path") # path to running videos (MP4)

# params with defaults
float(set_value("zoom_level",1))
PREDICT_INTERVAL = float(set_value("predict_interval",400.0))
SMOOTH_WINDOW = float(set_value("smooth_window",2.0))
DEFAULT_SPEED = float(set_value("default_speed",9.0))
MAX_SPEED = float(set_value("max_speed",20))
EXTRA_PIXEL_RATIO = float(set_value("extra_pixel_ratio",0.08))
SHIFT_CENTER_X = int(set_value("shift_center_x",0))
SHIFT_CENTER_Y = int(set_value("shift_center_y",0))

current_running_speed = -1 # global variable
running_speed_history = [] # keep track of previous speeds

light_green = QColor(144, 238, 144)
light_red = QColor(255, 182, 193)
white = QColor(255, 255, 255)

# FOR DEBUGGING AND DEVELOPMENT
import matplotlib.pyplot as plt
WEBCAM_VIDEO = "" # r"C:\Users\janne\Desktop\TreadmillApp\mydata\WIN_20210713_20_33_27_Pro.mp4"
DEBUG_SPEED_OVERRIDE = None #13
LOAD_MODEL = True # set false for faster loading in debugging

# correlation image registering
def register_image(image,offset_image):
    MAX_CORRECTION = int(0.10*min(image.shape[0],image.shape[1]))

    # pixel precision first
    shift, error, diffphase = phase_cross_correlation(image, np.fft.fft2(cv2.cvtColor(offset_image,cv2.COLOR_BGR2GRAY)),space='fourier')
    # subpixel precision
    # shift, error, diffphase = phase_cross_correlation(image, offset_image,
    #print(f"Detected subpixel offset (y, x): {shift}")
    if abs(shift[0])>MAX_CORRECTION:
        shift[0] = 0
    if abs(shift[1])>MAX_CORRECTION:
        shift[1] = 0

    offset_image = np.roll(offset_image,int(shift[0]),axis=0)
    offset_image = np.roll(offset_image,int(shift[1]),axis=1)
    #if abs(shift[0])+abs(shift[1])>0:
    #    print('fixed motion for x=%i, y=%i' % (shift[0],shift[1]))
    return offset_image

# this is the window that playes the selected video with dynamic playrate
class VideoWindow(QMainWindow):

    def __init__(self,initial_video = None,default_speed = 9.0,*args, **kwargs):
        super(VideoWindow, self).__init__(*args, **kwargs)

        self.is_paused = False
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

        self.vlcInstance = vlc.Instance('--video-on-top','--disable-screensaver','--high-priority','--loop','--playlist-autostart') # could put in instance parameters here
        self.videoPlayer = self.vlcInstance.media_player_new()
        self.videoPlayer.video_set_mouse_input(False)
        self.videoPlayer.video_set_key_input(False)
        self.videoPlayer.set_mrl("http://xxx.xxx.xxx.xxx", "network-caching=300")
        #self.videoPlayer.audio_set_mute(True)
        self.videoPlayer.audio_set_volume(50)
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

        #self.stopbutton = QPushButton("Stop",self)
        #self.hbuttonbox.addWidget(self.stopbutton)
        #self.stopbutton.clicked.connect(self.stop)

        self.timetxt = QLabel('0', self)
        self.timetxt.setFont(QFont('Times',11))
        self.timetxt.setAlignment(Qt.AlignCenter)
        self.hbuttonbox.addWidget(self.timetxt)

        self.positionslider = QSlider(Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.setMinimumWidth(500)
        self.positionslider.setTickInterval(1)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.set_position)

        self.timer = QTimer(self)
        self.timer.setInterval(500) # polling every 250ms
        self.media_duration=-1
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

        self.timetxt_remain = QLabel('0', self)
        self.timetxt_remain.setFont(QFont('Times',11))
        self.timetxt_remain.setAlignment(Qt.AlignCenter)
        self.hbuttonbox.addWidget(self.timetxt_remain)

        self.speedtxt = QLabel('0', self)
        self.speedtxt.setFont(QFont('Times',12))
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
        #if initial_video is not None and os.path.exists(initial_video):

        print('Loaded default media')
        self.media = vlc.Media(initial_video)
        # setting media to the media player
        self.videoPlayer.set_media(self.media)
        #self.vlcInstance.vlm_set_loop(initial_video, True) # didn't work!
        self.playbutton.setText("Pause")

        #self.videoPlayer.playButton()
        self.mainFrame.setLayout(t_lay_parent)

        self.min_playrate = float(config_data["main"]["min_playrate"])
        self.playrate = 1.0
        self.update_speed()
        self.videoPlayer.play()
        time.sleep(1)
        self.maxtime_ms = self.videoPlayer.get_length()

        self.videoFrame.setMouseTracking(True)
        self.videoFrame.mouseMoveEvent = self.mousemove
        self.last_mouse_move = time.time()

        self.show()

    def mousemove(self,e):
        self.last_mouse_move = time.time()

    def closeEvent(self, event):
        self.videoPlayer.stop()

    def resizeEvent(self, event):
        x = self.rect().getCoords()
        self.playbutton.move(10, x[3] - 35)
        #self.stopbutton.move(120, x[3] - 35)
        self.timetxt.move(105,x[3] - 35)
        self.timetxt.setFixedWidth(100)
        
        self.positionslider.move(200, x[3] - 35)
        self.positionslider.setFixedWidth(x[2]-515)
        
        self.timetxt_remain.move(x[2]-310, x[3] - 35)
        self.timetxt_remain.setFixedWidth(120)
        self.speedtxt.move(x[2] - 190, x[3] - 35)
        self.speedtxt.setFixedWidth(180)
        
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

        if DEBUG_SPEED_OVERRIDE is not None:
            self.playrate = max(self.min_playrate,DEBUG_SPEED_OVERRIDE/self.default_speed)
        else:
            if current_running_speed<0:
                self.speedtxt.setStyleSheet("color: red;  background-color: black")
            else:
                self.speedtxt.setStyleSheet("color: black;  background-color: white")
                self.playrate = max(self.min_playrate,current_running_speed/self.default_speed)

        self.speedtxt.setText('%2.1fkmh (%1.2f)' % (current_running_speed,self.playrate))

    def update_ui(self):
        """Updates the user interface"""

        # Set the slider's position to its corresponding media position
        # Note that the setValue function only takes values of type int,
        # so we must first convert the corresponding media position.
        #print('executed update_ui')
        now = time.time()
        if (now - self.last_mouse_move) > 4:
            self.videoFrame.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))
        else:
            self.videoFrame.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        pos = self.videoPlayer.get_position()
        media_pos = int(pos * 1000)
        self.positionslider.setValue(media_pos)

        if self.media_duration<=5:
            self.media_duration = self.videoPlayer.get_length()/1000

        remaining = self.media_duration*(1.0-pos)
        if self.media_duration > 10 and remaining<5:
            self.videoPlayer.set_position(0)
            return

        fps = -1
        try:
            fps = self.videoPlayer.get_fps()
        except:
            pass

        fps = fps if fps <150 and fps>5 else -1
        curtime = self.videoPlayer.get_time()
        self.timetxt.setText(self.ms_to_timestr(max(0,curtime))) # + " (%1.0f)" % fps )
        self.timetxt_remain.setText(self.ms_to_timestr(self.maxtime_ms - curtime))  # + " (%1.0f)" % fps )
        self.update_speed()
        self.videoPlayer.set_rate(self.playrate)
        #print("rate is %1.3f" % self.videoPlayer.get_rate())

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
        return "%02i:%02i:%02.0f" % (hours,minutes,sec)

    def play_pause(self):
        """Toggle play/pause status
        """
        if self.videoPlayer.is_playing():
            self.videoPlayer.pause()
            self.is_paused=True
            self.playbutton.setText("Play")
            self.timer.stop()
        else:
            self.videoPlayer.play()
            self.playbutton.setText("Pause")
            self.is_paused=False
            self.timer.start()

    def stop(self):
        """Stop player
        """
        self.videoPlayer.stop()
        self.playbutton.setText("Play")
    '''
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
    '''
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

        SCALE_UI = 1.30  # Change this value to adjust the scaling

        myGUI = self
        myGUI.setObjectName("myGUI")
        myGUI.resize(int(875 * SCALE_UI), int(461 * SCALE_UI))
        myGUI.setMaximumWidth(myGUI.width())
        myGUI.setMaximumHeight(myGUI.height())

        self.precision_group = QtWidgets.QButtonGroup(myGUI)
        self.precision0 = QtWidgets.QRadioButton(myGUI)
        self.precision0.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(280 * SCALE_UI), int(41 * SCALE_UI), int(31 * SCALE_UI)))
        self.precision0.setObjectName("precision0")
        self.precision1 = QtWidgets.QRadioButton(myGUI)
        self.precision1.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(310 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.precision1.setObjectName("precision1")
        self.precision2 = QtWidgets.QRadioButton(myGUI)
        self.precision2.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(330 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.precision2.setObjectName("precision2")
        self.precision3 = QtWidgets.QRadioButton(myGUI)
        self.precision3.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(350 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.precision3.setObjectName("precision3")

        self.zoomlevel = QtWidgets.QSlider(Qt.Horizontal, myGUI)
        self.zoomlevel.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(373 * SCALE_UI), int(41 * SCALE_UI), int(16 * SCALE_UI)))
        self.zoomlevel.setMinimum(1)
        self.zoomlevel.setMaximum(3)
        self.zoomlevel.setValue(int(float(config_data.get("main", "zoom_level"))))
        self.zoomlevel.setSingleStep(1)
        self.zoomlevel.setTickInterval(1)
        self.zoomlevel.setObjectName("zoomlevel")
        self.zoomlevel.valueChanged.connect(self.zoom_level_changed)

        self.label_zoom = QtWidgets.QLabel(myGUI)
        self.label_zoom.setGeometry(QtCore.QRect(int(770 * SCALE_UI), int(373 * SCALE_UI), int(35 * SCALE_UI), int(16 * SCALE_UI)))
        self.label_zoom.setObjectName("label_zoom")

        self.precision1.setChecked(1)
        self.precision_group.addButton(self.precision0, 0)
        self.precision_group.addButton(self.precision1, 1)
        self.precision_group.addButton(self.precision2, 2)
        self.precision_group.addButton(self.precision3, 3)
        self.precision_group.buttonClicked.connect(self.precision_event)

        self.digitcount_group = QtWidgets.QButtonGroup(myGUI)
        self.digitcount2 = QtWidgets.QRadioButton(myGUI)
        self.digitcount2.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(170 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.digitcount2.setObjectName("digitcount2")
        self.digitcount4 = QtWidgets.QRadioButton(myGUI)
        self.digitcount4.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(210 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.digitcount4.setObjectName("digitcount4")
        self.digitcount5 = QtWidgets.QRadioButton(myGUI)
        self.digitcount5.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(230 * SCALE_UI), int(41 * SCALE_UI), int(16 * SCALE_UI)))
        self.digitcount5.setObjectName("digitcount5")
        self.digitcount1 = QtWidgets.QRadioButton(myGUI)
        self.digitcount1.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(140 * SCALE_UI), int(41 * SCALE_UI), int(31 * SCALE_UI)))
        self.digitcount1.setObjectName("digitcount1")
        self.digitcount3 = QtWidgets.QRadioButton(myGUI)
        self.digitcount3.setGeometry(QtCore.QRect(int(800 * SCALE_UI), int(190 * SCALE_UI), int(41 * SCALE_UI), int(17 * SCALE_UI)))
        self.digitcount3.setObjectName("digitcount3")
        self.digitcount_group.buttonClicked.connect(self.digitcount_event)

        self.digitcount3.setChecked(1)
        self.digitcount_group.addButton(self.digitcount1, 1)
        self.digitcount_group.addButton(self.digitcount2, 2)
        self.digitcount_group.addButton(self.digitcount3, 3)
        self.digitcount_group.addButton(self.digitcount4, 4)
        self.digitcount_group.addButton(self.digitcount5, 5)

        self.label = QtWidgets.QLabel(myGUI)
        self.label.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(120 * SCALE_UI), int(51 * SCALE_UI), int(20 * SCALE_UI)))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(myGUI)
        self.label_2.setGeometry(QtCore.QRect(int(790 * SCALE_UI), int(260 * SCALE_UI), int(51 * SCALE_UI), int(20 * SCALE_UI)))
        self.label_2.setObjectName("label_2")

        self.regionselection = QtWidgets.QLabel(myGUI)
        self.regionselection.setGeometry(QtCore.QRect(int(500 * SCALE_UI), int(70 * SCALE_UI), int(150 * SCALE_UI), int(20 * SCALE_UI)))
        self.regionselection.setObjectName("regionselection")
        self.regionselection.setText("No ROI selected")

        self.lcdNumber = QtWidgets.QLCDNumber(myGUI)
        self.lcdNumber.setGeometry(QtCore.QRect(int(630 * SCALE_UI), int(15 * SCALE_UI), int(131 * SCALE_UI), int(31 * SCALE_UI)))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setStyleSheet("QLCDNumber { background-color: rgb(255, 255,255);color: rgb(0,255,0);font-weight: bold;}")

        self.frames_per_sec = QtWidgets.QLabel(myGUI)
        self.frames_per_sec.setGeometry(QtCore.QRect(int(660 * SCALE_UI), int(60 * SCALE_UI), int(100 * SCALE_UI), int(28 * SCALE_UI)))
        self.frames_per_sec.setText("0 frames/sec")

        self.medianbox = QtWidgets.QCheckBox(myGUI)
        self.medianbox.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(400 * SCALE_UI), int(81 * SCALE_UI), int(17 * SCALE_UI)))
        self.medianbox.setTristate(False)
        self.medianbox.setChecked(False)
        self.medianbox.setObjectName("medianbox")
        if int(config_data.get("main", "median_box")) == 1:
            self.medianbox.setChecked(True)
        else:
            self.medianbox.setChecked(False)

        self.tracking = QtWidgets.QCheckBox(myGUI)
        self.tracking.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(25 * SCALE_UI), int(81 * SCALE_UI), int(17 * SCALE_UI)))
        self.tracking.setTristate(False)
        self.tracking.setChecked(True)
        self.tracking.setObjectName("tracking")
        self.tracking.clicked.connect(self.start_camera)

        self.boxdetect = QtWidgets.QCheckBox(myGUI)
        self.boxdetect.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(420 * SCALE_UI), int(81 * SCALE_UI), int(17 * SCALE_UI)))
        self.boxdetect.setTristate(False)
        self.boxdetect.setObjectName("boxdetect")
        if int(config_data.get("main", "box_detect")) == 1:
            self.boxdetect.setChecked(True)
        else:
            self.boxdetect.setChecked(False)

        self.graphicsView = QtWidgets.QLabel(myGUI)
        self.graphicsView.setGeometry(QtCore.QRect(int(335 * SCALE_UI), int(100 * SCALE_UI), int(431 * SCALE_UI), int(351 * SCALE_UI)))
        self.graphicsView.setObjectName("graphicsView")

        self.videopathEdit = QtWidgets.QTextEdit(myGUI)
        self.videopathEdit.setGeometry(QtCore.QRect(int(90 * SCALE_UI), int(10 * SCALE_UI), int(421 * SCALE_UI), int(31 * SCALE_UI)))
        self.videopathEdit.setObjectName("textEdit")
        self.videopathEdit.setText(DEFAULT_PATH)
        self.videopathEdit.textChanged.connect(self.updatefiles)

        self.label_3 = QtWidgets.QLabel(myGUI)
        self.label_3.setGeometry(QtCore.QRect(int(550 * SCALE_UI), int(20 * SCALE_UI), int(71 * SCALE_UI), int(16 * SCALE_UI)))
        self.label_3.setObjectName("label_3")
        self.listWidget = QtWidgets.QListWidget(myGUI)
        self.listWidget.setGeometry(QtCore.QRect(int(20 * SCALE_UI), int(90 * SCALE_UI), int(311 * SCALE_UI), int(341 * SCALE_UI)))
        self.listWidget.setObjectName("listWidget")
        self.listWidget.itemClicked.connect(self.onClickedFile)
        self.listWidget.currentItemChanged.connect(self.onClickedFile)

        self.source_group = QtWidgets.QButtonGroup(myGUI)
        self.source0 = QtWidgets.QRadioButton(myGUI)
        self.source0.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(50 * SCALE_UI), int(71 * SCALE_UI), int(17 * SCALE_UI)))
        self.source0.setObjectName("source0")
        self.source1 = QtWidgets.QRadioButton(myGUI)
        self.source1.setGeometry(QtCore.QRect(int(780 * SCALE_UI), int(70 * SCALE_UI), int(71 * SCALE_UI), int(17 * SCALE_UI)))
        self.source1.setObjectName("source1")
        self.source_group.addButton(self.source0, 0)
        self.source_group.addButton(self.source1, 1)
        self.source_group.buttonClicked.connect(self.source_event)
        if int(config_data.get("main", "source")) == 0:
            self.source0.setChecked(True)
        else:
            self.source1.setChecked(True)

        self.label_4 = QtWidgets.QLabel(myGUI)
        self.label_4.setGeometry(QtCore.QRect(int(20 * SCALE_UI), int(10 * SCALE_UI), int(71 * SCALE_UI), int(16 * SCALE_UI)))
        self.label_4.setObjectName("label_4")
        self.playButton = QtWidgets.QPushButton(myGUI)
        self.playButton.setGeometry(QtCore.QRect(int(410 * SCALE_UI), int(60 * SCALE_UI), int(75 * SCALE_UI), int(23 * SCALE_UI)))  # left, top, width and height
        self.playButton.setObjectName("play")
        self.playButton.clicked.connect(self.play_movie)
        self.playButton.setEnabled(False)

        self.videospeedEdit = QtWidgets.QTextEdit(myGUI)
        self.videospeedEdit.setGeometry(QtCore.QRect(int(333 * SCALE_UI), int(60 * SCALE_UI), int(45 * SCALE_UI), int(23 * SCALE_UI)))
        self.videospeedEdit.setObjectName("videospeed")
        self.videospeedEdit.setText("")
        self.videospeedEdit.textChanged.connect(self.edited_speed)
        self.videospeedEdit.setEnabled(False)

        self.scoregoodbutton = QtWidgets.QPushButton(myGUI)
        self.scoregoodbutton.setGeometry(QtCore.QRect(int(380 * SCALE_UI), int(50 * SCALE_UI), int(25 * SCALE_UI), int(23 * SCALE_UI)))
        self.scoregoodbutton.setObjectName("scorebutton_good")
        self.scoregoodbutton.setText("")
        self.scoregoodbutton.setStyleSheet("background-color: white;")
        self.scoregoodbutton.clicked.connect(self.on_click_button_good)
        self.scoregoodbutton.setEnabled(False)

        self.scorebadbutton = QtWidgets.QPushButton(myGUI)
        self.scorebadbutton.setGeometry(QtCore.QRect(int(380 * SCALE_UI), int(75 * SCALE_UI), int(25 * SCALE_UI), int(23 * SCALE_UI)))
        self.scorebadbutton.setObjectName("scorebutton_bad")
        self.scorebadbutton.setText("")
        self.scorebadbutton.setStyleSheet("background-color: white;")
        self.scorebadbutton.clicked.connect(self.on_click_button_bad)
        self.scorebadbutton.setEnabled(False)
        #self.scoregoodEdit.textChanged.connect(self.edited_speed)
        #self.scoregoodEdit.setEnabled(False)

        self.currentvideoEdit = QtWidgets.QPlainTextEdit(myGUI)
        self.currentvideoEdit.setGeometry(QtCore.QRect(int(20 * SCALE_UI), int(60 * SCALE_UI), int(311 * SCALE_UI), int(21 * SCALE_UI)))
        self.currentvideoEdit.setObjectName("plainTextEdit")
        self.currentvideoEdit.setReadOnly(True)

        self.mouse_pos = None

        try:
            x1,y1,x2,y2,scale = [int(x) for x in config_data.get("main", "roi").split(",")]
            assert abs(x1-x2)>2 and abs(y1-y2)>2,"Bad ROI in config!"
            self.coordinate1 = (x1,y1,scale)
            self.coordinate2 = (x2,y2,scale)
            self.selected_ROI = True
            print("predefined ROI loaded")
        except:
            self.selected_ROI = False
            self.coordinate1 = None
            self.coordinate2 = None

        self.frame_shape = None
        self.frame_scaling = None

        self.retranslateUi(myGUI)
        QtCore.QMetaObject.connectSlotsByName(myGUI)

        self.filelist = None
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

    def on_click_button_good(self):
        if self.chosen_file is not None:
            if self.chosen_file["score"]==1:
                self.chosen_file["score"]=0
                self.scoregoodbutton.setStyleSheet("background-color: white;")
            else:
                self.chosen_file["score"] = 1
                self.scoregoodbutton.setStyleSheet("background-color: green;")
            self.scorebadbutton.setStyleSheet("background-color: white;")
            self.update_filedata()
            self.save_configs()

    def on_click_button_bad(self):
        if self.chosen_file is not None:
            if self.chosen_file["score"] == -1:
                self.chosen_file["score"] = 0
                self.scorebadbutton.setStyleSheet("background-color: white;")
            else:
                self.chosen_file["score"] = -1
                self.scorebadbutton.setStyleSheet("background-color: red;")
            self.scoregoodbutton.setStyleSheet("background-color: white;")
            self.update_filedata()
            self.save_configs()
    def edited_speed(self):
        pass

    def set_running_speed(self,val):
        self.lcdNumber.setNumDigits(self.digitcount+1)
        self.lcdNumber.display(val)

    def digitcount_event(self,e):
        self.digitcount = self.digitcount_group.id(e)
        print("changed digitcount to %i" % self.digitcount)

    def source_event(self,e):
        print("changed source to %i" % self.source_group.checkedId())
        self.start_camera()

    def precision_event(self,e):
        self.precision = self.precision_group.id(e)
        print("changed precision to %i" % self.precision)

    def mousemove(self,e):
        if self.frame_shape is not None:
            #zoom_level = self.zoomlevel.value()
            x = e.x()
            y = e.y()
            #print("mouse coord current: x=%i y=%i" % (int(x), int(y)))
            self.mouse_pos = (x, y)

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

    # webcam frame click release, gets coordinate 2
    def mouseup(self,e):
        # Record ending (x,y) coordintes on left mouse bottom release
        pass
        '''
        if e.button()==1 and not(self.selected_ROI) and self.frame_shape is not None:
            x = e.x()
            y = e.y()
            self.coordinate2 = (x,y,self.zoomlevel.value())
            print("mouse coord 2 (x, y) = (%i, %i)" % (self.coordinate2[0], self.coordinate2[1]))
            self.selected_ROI = True

            # update config file
            config_data["main"]["roi"] = "%i,%i,%i,%i,%i" % (self.coordinate1[0],self.coordinate1[1],self.coordinate2[0],self.coordinate2[1],self.zoomlevel.value())
            with open(CONFIG_FILE, 'w',encoding="UTF-8") as configfile:
                config_data.write(configfile)
        '''
    def zoom_level_changed(self,value):
        config_data["main"]['zoom_level'] = str(value)
        self.save_configs()

    def save_configs(self):
        with open(CONFIG_FILE, 'w', encoding="UTF-8") as configfile:
            config_data.write(configfile)
    # clicked webcam frame
    def mousedown(self, e):
        # Record starting (x,y) coordinates on left mouse button click
        if e.button()==1 and not(self.selected_ROI) and self.frame_shape is not None:
            x = e.x()
            y = e.y()
            if self.coordinate1 is None:
                self.coordinate1 = (x, y,self.zoomlevel.value())
                self.coordinate2 = None
                self.selected_ROI = False
                print("mouse coord 1 (x, y) = (%i, %i)" % (self.coordinate1[0], self.coordinate1[1]))
            else:
                self.coordinate2 = (x, y, self.zoomlevel.value())
                print("mouse coord 2 (x, y) = (%i, %i)" % (self.coordinate2[0], self.coordinate2[1]))
                self.selected_ROI = True
                # update config file
                config_data["main"]["roi"] = "%i,%i,%i,%i,%i" % (
                    self.coordinate1[0], self.coordinate1[1], self.coordinate2[0], self.coordinate2[1],
                    self.zoomlevel.value())
                self.save_configs()

        # Clear drawing boxes on right mouse button click
        elif e.button()==2:
            self.selected_ROI = False
            self.coordinate1 = None
            self.coordinate2 = None
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
            current_running_speed=DEFAULT_SPEED # self.defaultspeed
            self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
            self.frames_per_sec.setText("-")

    # function to capture and analyze webcam stream (running as separate thread)
    def camera_update(self,device):
        global running_speed_history,current_running_speed
        INITIAL_FRAMES = 8 # initial frames to set-up template frame

        DEBUGGING_MODE = False
        if len(WEBCAM_VIDEO)>0:
            DEBUGGING_MODE = True
            assert os.path.exists(WEBCAM_VIDEO),"Webcam video not found!"
            print("!! Loading webcam video instead of actual stream (DEBUGGING)!!")
            self.capture = cv2.VideoCapture(WEBCAM_VIDEO)
            old_images= [None,None,None]
            image_num = 0
            debug_fig,debug_ax = plt.subplots(1, 3, figsize=[3 * 3,3])
            speed_estimation_file = open("speed_predictions_debug.csv","w")
        else:
            self.capture = cv2.VideoCapture(device,cv2.CAP_DSHOW)
        old_device = device
        rectangles = None
        last_prediction_time = time.time() - PREDICT_INTERVAL

        print("Starting video capture")
        init_run=0
        template_frame = 0 # this will be FFT transformed
        zoom_cut = {}
        zoom_multiplier = {}
        old_ROI=None

        # convert view coordinates to actual coordinates in acquired frame. Need to take into account padding and zooming!
        def view_to_frame(coordinate1,coordinate2):
            # remove padded pixels outside frame, convert to zoom=1 scale and add cropped pixels
            x1 = (coordinate1[0]-self.frame_padding["width"]) * zoom_multiplier[coordinate1[2]] + zoom_cut[coordinate1[2]]["view"]["width"]
            y1 = (coordinate1[1]-self.frame_padding["height"]) * zoom_multiplier[coordinate1[2]] + zoom_cut[coordinate1[2]]["view"]["height"]

            x2 = (coordinate2[0]-self.frame_padding["width"]) * zoom_multiplier[coordinate1[2]] + zoom_cut[coordinate1[2]]["view"]["width"]
            y2 = (coordinate2[1]-self.frame_padding["height"]) * zoom_multiplier[coordinate1[2]] + zoom_cut[coordinate1[2]]["view"]["height"]

            x1 = min(x1, x2)
            x2 = max(x1, x2)
            y1 = min(y1, y2)
            y2 = max(y1, y2)

            # scale coordinates to frame
            x1 = int(x1 * self.frame_scaling["width"])
            y1 = int(y1 * self.frame_scaling["height"])
            x2 = int(x2 * self.frame_scaling["width"])
            y2 = int(y2 * self.frame_scaling["height"])            

            return x1, y1, x2, y2

        frame_num = 0
        while True:
            if self.capture.isOpened():
                if self.source_group.checkedId() != old_device or not(self.tracking.isChecked()):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    current_running_speed = self.defaultspeed
                    self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
                    return
                # Read frame
                #if init_run >= 5:
                #    frame = template_frame_real
                #else:
                (status, frame) = self.capture.read()
                
                #print('frame.shape = %s' % str(frame.shape))
                frame = np.roll(frame,SHIFT_CENTER_X,axis=1)
                frame = np.roll(frame,SHIFT_CENTER_Y,axis=0)

                if frame is None or not(isinstance(frame,np.ndarray)):
                    print('obtained frame is not valid (type %s)!' % str(type(frame)))
                    continue

                frame_num+=1

                if init_run<INITIAL_FRAMES:

                    image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)
                    pixmap = pixmap.scaled(self.graphicsView.width(), self.graphicsView.height(), QtCore.Qt.KeepAspectRatio)

                    init_run+=1
                    template_frame += frame.astype(float)/INITIAL_FRAMES
                    if init_run==INITIAL_FRAMES:
                        #template_frame_real = template_frame.astype(np.uint8)
                        #print('template_frame.shape = %s' % str(template_frame.shape))
                        
                        template_frame = np.fft.fft2(cv2.cvtColor(template_frame.astype(np.uint8),cv2.COLOR_BGR2GRAY))

                        self.frame_shape = {"width":frame.shape[1],"height":frame.shape[0]}  # put this here to make sure we have also padding defined
                        self.frame_scaling = {"width":self.frame_shape["width"]/pixmap.width(),"height":self.frame_shape["height"]/pixmap.height()}
                        self.frame_padding = {"width":int((self.graphicsView.width() - pixmap.width()) / 2),"height":int((self.graphicsView.height() - pixmap.height()) / 2)}

                        for zoom_level in [1,2,3]:
                            zoom_cut[zoom_level] = {}
                            zoom_cut[zoom_level]["orig"] = {
                                "width":((self.frame_shape["width"] / 2) * (zoom_level-1) / 4),
                                "height":((self.frame_shape["height"] / 2) * (zoom_level-1) / 4),
                            }
                            zoom_cut[zoom_level]["view"] = {
                                "width":int(zoom_cut[zoom_level]["orig"]["width"]/self.frame_scaling["width"]),
                                "height":int(zoom_cut[zoom_level]["orig"]["height"]/self.frame_scaling["height"]),
                            }
                            zoom_cut[zoom_level]["orig"]["width"]=int(zoom_cut[zoom_level]["orig"]["width"])
                            zoom_cut[zoom_level]["orig"]["height"] = int(zoom_cut[zoom_level]["orig"]["height"])

                            zoom_multiplier[zoom_level] = 1 - (zoom_level-1)/4

                        self.graphicsView.setMouseTracking(True)
                        self.graphicsView.mouseMoveEvent = self.mousemove
                        self.graphicsView.mousePressEvent = self.mousedown
                        self.graphicsView.mouseReleaseEvent = self.mouseup

                else:
                    frame = register_image(template_frame,frame)

                    now = time.time()
                    if self.selected_ROI:
                        if old_ROI is None:
                            x1,y1,x2,y2 = view_to_frame(self.coordinate1,self.coordinate2)
                            # scale coordinated to actual frame
                            ROI = [x1,y1,x2,y2,self.digitcount] # [x1,y1,x2,y2,digit_count]

                            # (x2-x1) = width
                            if (x2-x1)<40 or (y2-y1)<20 or (ROI[2]>frame.shape[1]) or (ROI[3]>frame.shape[0]) or (self.coordinate1[2] != self.coordinate2[2]):
                                print("Bad ROI, resetting")
                                self.selected_ROI = False
                                self.coordinate1 = None
                                self.coordinate2 = None
                                continue
                            print("ROI coordinates: x1=%i, y1=%i, x2=%i, y2=%i" % (x1,y1,x2,y2))
                            old_ROI=ROI
                        else:
                            ROI = old_ROI

                        # if enough time passed, analyze digits
                        if (now - last_prediction_time)*1000 > PREDICT_INTERVAL:
                            predicted_digits,predicted_prob,rectangles,raw_frames = predict_digit(frame, ROI,self.medianbox.isChecked(),self.boxdetect.isChecked())
                            if predicted_digits is None:
                                # failed, just use last valid prediction
                                predicted_speed = current_running_speed
                                predicted_prob = 0
                            else:
                                # convert individual digits to float using selected precision
                                if 0:#DEBUGGING_MODE:
                                    for k in range(3):
                                        if old_images[k] is not None:
                                            old_images[k].set_data(raw_frames[k])
                                        else:
                                            old_images[k]=debug_ax[k].imshow(raw_frames[k])
                                        if np.random.rand()<1/10 and predicted_digits[k]>-1:
                                            image_num += 1
                                            plt.imsave("%i.png" % (image_num), raw_frames[k])
                                predicted_digits_str = "".join([str(x) if x > -1 else "0" for x in predicted_digits])
                                predicted_digits_str = predicted_digits_str[0:(len(predicted_digits)-self.precision)] + "." + predicted_digits_str[-self.precision:]
                                predicted_speed = float(predicted_digits_str)

                            timestamp = now - self.starttime
                            if DEBUGGING_MODE:
                                speed_estimation_file.writelines("frame=%i, predicted_speed=%f, prob=%f\n" % (frame_num,predicted_speed,predicted_prob))
                                speed_estimation_file.flush()

                            running_speed_history.append((timestamp, predicted_speed,predicted_prob))
                            # a function to obtain speed, could include some fancy smoothing and error-checking
                            current_running_speed = speed_estimator(timestamp,running_speed_history)

                            # only keep last 100 measurements
                            if len(running_speed_history)>50:
                                running_speed_history = running_speed_history[-50:]

                            self.lcdNumber.display("{1:,.{0}f}".format(self.precision,current_running_speed))
                            self.frames_per_sec.setText("%.2f frames/sec" % (1.0 / (max(0.0001,time.time()-now))))
                            last_prediction_time = now

                        frame = cv2.rectangle(frame, (ROI[0],ROI[1]),(ROI[2],ROI[3]), (0, 255, 0),3)
                        if rectangles is not None:
                            for k,rect in enumerate(rectangles):
                                col = (0, 200,200) if (k%2==0) else (200,0,200)
                                frame = cv2.rectangle(frame,
                                                    (rect[0],rect[2]),
                                                    (rect[1],rect[3]),
                                                    col,2)

                        self.regionselection.setText("ROI (%i,%i,%i,%i)" % (x1,y1,x2,y2))

                        #for r in self.digit_ROIs:
                        #    self.frame = cv2.rectangle(self.frame, (r[1][0]+x1,r[1][1]+y1), (r[1][2]+x1,r[1][3]+y1), (0,50,255), 2)
                    elif self.coordinate1 is not None:
                        # obtain coordinated in frame view
                        #self.mouse_pos = self.coordinate1
                        x1, y1, x2, y2 = view_to_frame(self.coordinate1, self.mouse_pos)
                        frame = cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
                        rectangles = None
                        old_ROI = None

                    zoom_level = self.zoomlevel.value()
                    if zoom_level > 1:
                        frame = frame[zoom_cut[zoom_level]["orig"]["height"]:-zoom_cut[zoom_level]["orig"]["height"], zoom_cut[zoom_level]["orig"]["width"]:-zoom_cut[zoom_level]["orig"]["width"], :]

                    image = QtGui.QImage(frame.data.tobytes(), frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    pixmap = QtGui.QPixmap.fromImage(image)
                    pixmap = pixmap.scaled(self.graphicsView.width(), self.graphicsView.height(), QtCore.Qt.KeepAspectRatio)

                self.graphicsView.setPixmap(pixmap)

                key = cv2.waitKey(1)

            else:

                print("Capture terminated!")
                self.capture.release()
                cv2.destroyAllWindows()
                current_running_speed = self.defaultspeed
                self.lcdNumber.display("{1:,.{0}f}".format(self.precision, current_running_speed))
                break

    def updatefiles(self):
        # read all files with specific extension
        ROOT_PATH = self.videopathEdit.toPlainText() + os.sep
        files = glob.glob(ROOT_PATH + "*.mp4")
        files += glob.glob(ROOT_PATH + "*.webm")
        files += glob.glob(ROOT_PATH + "*.mkv")
        files += glob.glob(ROOT_PATH + "*.avi")
        files = [{"full_filename":x,"filename":x.replace(ROOT_PATH, ''),"size":os.path.getsize(x),"rate":DEFAULT_SPEED,"duration":-1,'score':0} for x in files]

        # update config file
        count = 0
        old_files = {k:config_data["files"].get(k).split("|") for k in list(config_data["files"])} # separator is pipe |
        for k,old_file in old_files.items():
            for f in files:
                if int(old_file[1])==f["size"]:
                    try:
                        f["rate"] = float(old_file[2])
                        f["duration"] = float(old_file[3])                
                        f['score'] = float(old_file[4])
                        count+=1
                    except:
                        print('failed to read old parameters for video "%s"' % old_file[0])

        print("loaded %i videos with %i new ones" % (len(files),len(files)-count))

        # count the number of frames
        print("computing playtime")
        k = 0
        for f in files:
            if f["duration"]==-1:
                k+=1
                try:
                    video_obj = cv2.VideoCapture(f["full_filename"])
                    frames = video_obj.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = int(video_obj.get(cv2.CAP_PROP_FPS))
                    # calculate dusration of the video
                    seconds = int(frames / fps)
                    f["duration"] = seconds/60.0
                except:
                    f["duration"] = 35.0
                    print("!! failed to extract video duration for file %s !!" % f["filename"])
            f["playtime"] = f["duration"]*f["rate"]/DEFAULT_SPEED

        files.sort(key= lambda x:x["playtime"], reverse=True) # sort by modified date

        self.filelist = {}
        self.listWidget.clear()
        for f in files:
            label = '%imin - %s' % (round(f["playtime"]),f["filename"])
            self.filelist[label] = f
            item = QListWidgetItem(label)
            if f['score']==1:
                item.setBackground(QBrush(light_green))
            elif f['score']==-1:
                item.setBackground(QBrush(light_red))
            else:
                item.setBackground(QBrush(white))
            self.listWidget.addItem(item)

        #print("saving configs")
        #config_data["files"] = {i:":".join([f["filename"],str(f["size"]),str(f["rate"]),str(round(f["duration"],3))]) for i,f in enumerate(files)}
        #with open(CONFIG_FILE, 'w',encoding="UTF-8") as configfile:
        #    config_data.write(configfile)

    def onClickedFile(self,item=None):
        if item is None:
            item = self.currentItem()
        self.currentvideoEdit.setPlainText(item.text())
        self.playButton.setEnabled(True)
        self.videospeedEdit.setEnabled(True)
        self.scorebadbutton.setEnabled(True)
        self.scoregoodbutton.setEnabled(True)

        #saved_files = {k:config_data["files"].get(k).split(":") for k in list(config_data["files"])}
        selected_file = self.filelist[item.text()]
        self.chosen_file = selected_file
        #for k,file in saved_files.items():
        #    if self.chosen_file[1] in file[0]:
        self.defaultspeed = float(selected_file["rate"])
        self.videospeedEdit.setText(str(self.defaultspeed))

        score = float(selected_file["score"])
        self.scoregoodbutton.setStyleSheet("background-color: white;")
        self.scorebadbutton.setStyleSheet("background-color: white;")
        if score==1:
            self.scoregoodbutton.setStyleSheet("background-color: green;")
        if score==-1:
            self.scorebadbutton.setStyleSheet("background-color: red;")
    def retranslateUi(self, myGUI):
        _translate = QtCore.QCoreApplication.translate
        myGUI.setWindowTitle(_translate("myGUI", "TreadmillApp"))
        self.precision0.setText(_translate("myGUI", "0"))
        self.precision1.setText(_translate("myGUI", "1"))
        self.precision2.setText(_translate("myGUI", "2"))
        self.precision3.setText(_translate("myGUI", "3"))
        self.digitcount2.setText(_translate("myGUI", "2"))
        self.digitcount4.setText(_translate("myGUI", "4"))
        self.digitcount5.setText(_translate("myGUI", "5"))
        self.digitcount1.setText(_translate("myGUI", "1"))
        self.digitcount3.setText(_translate("myGUI", "3"))
        self.label.setText(_translate("myGUI", "digit count"))
        self.label_2.setText(_translate("myGUI", "precision"))
        self.medianbox.setText(_translate("myGUI", "median box"))
        self.tracking.setText(_translate("myGUI", "tracking"))
        self.label_zoom.setText(_translate("myGUI", "zoom"))
        self.boxdetect.setText(_translate("myGUI", "box detect"))
        self.label_3.setText(_translate("myGUI", "current speed"))
        self.source0.setText(_translate("myGUI", "source 0"))
        self.source1.setText(_translate("myGUI", "source 1"))
        self.label_4.setText(_translate("myGUI", "current path"))
        self.playButton.setText(_translate("myGUI", "Play selected"))

    def update_filedata(self):
        if self.chosen_file is not None:
            old_files = {k:config_data["files"].get(k).split("|") for k in list(config_data["files"])}
            found=False
            new_data_entry = "|".join([
                self.chosen_file["filename"],
                str(self.chosen_file["size"]),
                str(self.chosen_file["rate"]),
                str(round(self.chosen_file["duration"], 3)),
                str(self.chosen_file["score"]),
            ])
            # is the file already in database? If not, add.
            for k,old_file in old_files.items():
                if int(old_file[1])==self.chosen_file['size']:
                    config_data["files"][k] = new_data_entry
                    found=True
                    break
            if found is False:
                config_data["files"][str(len(config_data["files"]))] = new_data_entry

            for index in range(self.listWidget.count()):
                item = self.listWidget.item(index)
                label = item.text()
                itemdata = self.filelist[label]
                if itemdata['full_filename']==self.chosen_file['full_filename']:
                    self.filelist[label]['score']=self.chosen_file['score']
                    if self.chosen_file['score'] == 1:
                        item.setBackground(QBrush(light_green))
                    elif self.chosen_file['score'] == -1:
                        item.setBackground(QBrush(light_red))
                    else:
                        item.setBackground(QBrush(white))

    # play selected video
    def play_movie(self):
        # first set and save default speed
        self.defaultspeed = float(self.videospeedEdit.toPlainText())
        for k,f in self.filelist.items():
            if f["size"] == self.chosen_file['size']:
                f["rate"] = self.defaultspeed
                break

        self.update_filedata()

        print("writing updated rate (%.3f)" % self.defaultspeed)
        self.save_configs()

        # finally play video in separate window
        self.videowindow = VideoWindow(self.chosen_file['full_filename'],self.defaultspeed)

    def closeEvent(self, event):
        if self.videowindow is not None:
            self.videowindow.close()

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    sorted_indices = np.argsort(data)
    data, weights = data[sorted_indices], weights[sorted_indices]
    cumulative_weights = np.cumsum(weights)
    midpoint = 0.5 * cumulative_weights[-1]
    if any(weights > midpoint):
        return (data[weights == np.max(weights)])[0]
    median_idx = np.searchsorted(cumulative_weights, midpoint)
    if cumulative_weights[median_idx] == midpoint:
        return np.mean(data[median_idx:median_idx + 2])
    return data[median_idx]
    
# function to compute speed from current and historical measurements
def speed_estimator(current_time,running_speed_history):
    prev_speeds = []
    prev_probs = []
    for x in reversed(running_speed_history):
        if (current_time - x[0])<SMOOTH_WINDOW:
            prev_speeds.append(x[1])
            prev_probs.append(x[2])
        else:
            break
    if len(prev_speeds)>5:
        display_speed = weighted_median(prev_speeds,prev_probs)
    else:
        display_speed = prev_speeds[0]
    display_speed = max(0.001, min(MAX_SPEED,display_speed))
    return display_speed

# given frame and ROI, return sub-images of digits
def image_preprocessor(img,ROI,use_prev_box,use_detection,prev_boxes=None):

    if prev_boxes is None:
        prev_boxes=[]
    ROI_w = ROI[2] - ROI[0]
    ROI_h = ROI[3] - ROI[1]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ROI_img = img[ROI[1]:(ROI[3] + 1), ROI[0]:(ROI[2] + 1), :]
    gray = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # find contours

    if use_detection:
        contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  #
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [x for x in boxes if x[2] > ROI_img.shape[1] * 0.70 and x[3] > ROI_img.shape[0] * 0.70]
        boxes = [x for x in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)]

    add_to_hist = True
    if not(use_detection) or len(boxes) == 0:
        boxes = [[0,0,ROI_w,ROI_h]]
        add_to_hist=False

    box = boxes[0]
    box = [max(0, box[0]), box[1], min(ROI_w, box[2]), box[3]] # x,y,w,h  ??

    if use_detection and add_to_hist:
        prev_boxes.append(box)
        if len(prev_boxes)>50:
            prev_boxes=prev_boxes[-50:]
    if use_detection and use_prev_box:
        box = [int(np.median([x[k] for x in prev_boxes])) for k in range(4)]

    dx = int(np.round((box[2])/ROI[-1])) # width of one box
    extra_pixels_x = int(np.ceil(dx*(EXTRA_PIXEL_RATIO)))
    extra_pixels_y = 1# int(np.ceil(box[3]*(EXTRA_PIXEL_RATIO)))

    digits = []
    rectangles = []
    for part in range(ROI[-1]):
        rectangles.append([
            ROI[0] + ROI_w - (box[0] + (dx * (part + 1))) - extra_pixels_x,
            ROI[0] + ROI_w - (box[0]+(dx * part)) + extra_pixels_x,
            ROI[1] + ROI_h-box[3] - extra_pixels_y,
            ROI[3] - box[1] + extra_pixels_y])
        sub_img = img[
                  (ROI[1]+box[1]-extra_pixels_y):(ROI[1]+box[1] + box[3]+extra_pixels_y+1),
                  (ROI[0]+box[0]+dx * part - extra_pixels_x):(ROI[0]+dx*(part + 1)+extra_pixels_x+1),
                  :]
        sub_img = np.flip(sub_img , axis=0)
        sub_img = np.flip(sub_img , axis=1)
        digits.append(sub_img)
    digits = list(reversed(digits))
    rectangles = list(reversed(rectangles))

    return digits,prev_boxes,rectangles

# resize sub-image with optional padding
def resize_image(img, size=(28,28),PAD = 0):
    # size = h,w
    interpolation = cv2.INTER_LINEAR
    size = size[0]-PAD,size[1]-PAD
    h, w = img.shape[:2]
    aspect_ratio = h/w
    new_aspect_ratio = size[0]/size[1]

    median_color = [int(x) for x in (0.5*np.median(img, axis=(0, 1)))]

    if abs(new_aspect_ratio - aspect_ratio)<1e-6:
        new_im = cv2.resize(img, size, interpolation)
        #new_im = cv2.copyMakeBorder(mask,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
        assert new_im.shape == IMG_SIZE
        return new_im
    elif new_aspect_ratio<aspect_ratio: # new image is too narrow, so need padding to width
        tmp_size = [size[0],int(np.round(size[0]/aspect_ratio))]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[1]-new_im.shape[1]
        new_im = cv2.copyMakeBorder(new_im,0,0,int(P/2),0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0, 0,0,size[1]-new_im.shape[1], cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
    else: # new image is too wide, so need padding to height
        tmp_size = [int(np.round(size[1]*aspect_ratio)),size[1]]
        new_im = cv2.resize(img,[tmp_size[1],tmp_size[0]],interpolation)
        P = size[0]-new_im.shape[0]
        new_im = cv2.copyMakeBorder(new_im,int(P/2),0,0,0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, right
        new_im = cv2.copyMakeBorder(new_im, 0,size[0]-new_im.shape[0],0,0, cv2.BORDER_CONSTANT, value=median_color)  # top, bottom, left, righ
    #new_im = cv2.copyMakeBorder(new_im,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right

    assert new_im.shape[:2] == IMG_SIZE,"BUG: Resized image has incorrect shape (%s)!" % str(new_im.shape)

    return new_im

# feed processed images to the predictor model
prev_boxes = []
predictor_model = None

def predict_digit(frame,ROI,use_prev_box,use_detection,DEBUG=False):
    global prev_boxes,predictor_model
    if predictor_model is None:
        return None,None,None,None
    frame_digits, prev_boxes, rectangles = image_preprocessor(frame,ROI,use_prev_box,use_detection,prev_boxes=prev_boxes)
    if len(frame_digits) != ROI[-1]:
        print("Failed to find individual digits (found %i digits our of required %i)!" % (len(frame_digits),ROI[-1]))
        return None,None,None,None
    if USE_PIL_IMAGE:
        frame_digits = [Image.fromarray(x.astype('uint8'), 'RGB') for x in frame_digits]
    else:
        raw_frame_digits = [resize_image(x, IMG_SIZE) for x in frame_digits]
        frame_digits = np.stack(raw_frame_digits, axis=0).astype(np.float32) / 255.0
    digits,prob = predictor_model.predict(frame_digits)
    prob = np.prod(prob)
    #digits = np.argmax(digits, 1)
    #digits[digits == 10] = -1

    return digits,prob,rectangles,frame_digits

if __name__ == "__main__":
    if 0:
        app = QtWidgets.QApplication(sys.argv)
        ui = VideoWindow('',4.0)
        sys.exit(app.exec_())
    else:
        USE_PIL_IMAGE = False
        if LOAD_MODEL:
            # load and test predictor model before starting the app
            print("Loading recognition model... ", end="")
            if config_data.get("main", "model_type") == 'tensorflow':
                predictor_model = load_model(DIGIT_MODEL)
                IMG_SIZE = predictor_model.layers[1].input_shape[1:3]
                print("making dummy prediction to initialize predictor... ", end='')
                predictor_model.predict(np.random.rand(10, IMG_SIZE[0], IMG_SIZE[1], 3))
            elif config_data.get("main", "model_type") == 'pytorch':
                USE_PIL_IMAGE = True
                device = torch.device('cpu')
                predictor_model = torch.load(DIGIT_MODEL, map_location=device)
                predictor_model.set_device(device)
                IMG_SIZE = [42,61]
                print("making dummy prediction to initialize predictor... ", end='')
                width, height = 46, 68
                array = np.random.rand(height, width, 3) * 255  # Random values between 0 and 255
                img = Image.fromarray(array.astype('uint8'), 'RGB')
                output,prob = predictor_model.predict([img])
            assert IMG_SIZE[0]>30 and IMG_SIZE[1]>30,"BAD IMAGE SIZE!"
            print("done")

        print("Starting application")
        app = QtWidgets.QApplication(sys.argv)
        ui = MainWindow()
        #QtCore.QTimer.singleShot(0,ui.close)
        sys.exit(app.exec_())


