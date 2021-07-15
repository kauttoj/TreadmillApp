# TreadmillApp

This repo contains codes for a Python-based video player with dynamic video playrate. Playrate is obtained from an external source, which in my case is a treadmill. The speed of the treadmill is obtained using a webcam and a bunch of machine vision and machine learning algorithms. I have trained a Keras-based neural network model to classify digits of my treadmill.

Main GUI window:
![TreadmillApp](https://user-images.githubusercontent.com/17804946/125808951-2df3f8c5-969e-4f34-a2e7-32d81a7e7510.png)

Video GUI window:
![TreadmillApp1](https://user-images.githubusercontent.com/17804946/125809859-65d4e50f-2015-4772-9aa5-91346724c51f.png)

Requirements:

	-Python 3
	-OpenCV
	-Keras
	-PyQT5
	-Webcam
	-VLC with Python support
	-At least one virtual running/walking/cycling video in MP4 format

Files:

	digit_recognition_model_development.py - codes to create a dataset and train a model for digit recognition
	TreadmillApp.py - actual codes to use trained model and show videos
	config_params.cfg - a config file to set parameters of the application

The idea of this app to be able to use free virtual running videos (see, e.g., Youtube) and dynamically change the video playrate depending on your actual speed on treadmill or spinning bike. This makes training more interactive and fun. Although there is nothing too complicated in this process, I didn't find any free applications that allow you to use your _own_ videos and real speed measurements. The best commercial software (e.g., Zwift) require montly fees, which I don't like. Hence I decided to try and make my own app for this purpose.

This is an early version without much polishing. This particular version is fine-tuned for my own treadmill, hence you probably need to modify the code/parameters and train a new model that works with your treadmill/bike. Note that instead of measuring speed using a webcam and machine vision, you could might use a speedometer, which is probably more simple and less error-prone. All you need to do is to make a function (and start a thread) that updates the global variable "current_running_speed" which is used to compute dynamic playrate.

15.7.2021 (initial version)
