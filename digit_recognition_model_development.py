# https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1

import tempfile
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import cv2
import pandas as pd
import time

save_file = "digit_recognition_model_ver5"
TRAIN_DATA = ['xxxxx','xxxxxxxx']

LOAD_OLD_MODEL = 0
NB_CHANNELS = 3
IMG_SIZE = (52,40) # Y, X
PAD = 4
MODE = 2 # 0 = prepare training data, 1 = train, 2 = test with video
MODEL_TYPE = 3 # neural network construct

# given frame and ROI, return sub-images of digits
def image_preprocessor(img,ROI,prev_boxes=None):
    if prev_boxes is None:
        prev_boxes=[]
    ROI_w = (ROI[2] + 1) - ROI[0]
    ROI_h = (ROI[3] + 1) - ROI[1]
    img = img[ROI[1]:(ROI[3] + 1), ROI[0]:(ROI[2] + 1), :]
    img = np.flip(img, axis=0)
    img = np.flip(img, axis=1)
    # print("image size = ",str(img.shape))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # find contours
    contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  #
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [x for x in boxes if x[2] > img.shape[1] * 0.70 and x[3] > img.shape[0] * 0.70]
    add_to_hist = True
    if len(boxes) == 0:
        extra = 0
        boxes = [[extra,extra,ROI_w-2*extra,ROI_h-2*extra]]
        add_to_hist=False
    boxes = [x for x in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)]
    box = boxes[0]
    if add_to_hist:
        prev_boxes.append(box)
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

# resize sub-image with padding
def resize_image(img, size=(28,28),PAD = 0):
    # size = h,w
    interpolation = cv2.INTER_AREA
    size = size[0]-PAD,size[1]-PAD
    extra_pad = int(PAD / 2)
    h, w = img.shape[:2]
    aspect_ratio = h/w
    new_aspect_ratio = size[0]/size[1]
    c = img.shape[2] if len(img.shape)>2 else 1
    if h == w:
        mask = cv2.resize(img, size, interpolation)
        new_im = cv2.copyMakeBorder(mask,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
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
    new_im = cv2.copyMakeBorder(new_im,extra_pad,extra_pad,extra_pad,extra_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # top, bottom, left, right
    assert new_im.shape[:2] == IMG_SIZE,"resized image incorrect shape (%s)!" % str(new_im.shape)
    return new_im

if MODE == 0:

    FOLDER = r'D:\JanneK\Documents\git_repos\MyMediaPlayer-main\mydata\raw_frames2'
    images = [cv2.imread(FOLDER + os.sep + file) for file in os.listdir(FOLDER) if file.endswith(".png")]
    images = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in images]

    OUTPUT = FOLDER + os.sep + "digits" + os.sep

    image_ROIs = [(485, 355, 612, 419, 3), (630, 357, 797, 417, 4), (357, 458, 520, 522, 4)]  # x1,y1,x2,y2
    image_ROIs += [(482, 353, 610, 423, 3), (634, 359, 795, 413, 4)]  # x1,y1,x2,y2

    image_ROIs = [(470, 191, 599, 258, 3), (615, 192, 781, 256, 4)]

    count = 0
    ALL_images = []

    for ind_roi, ROI in enumerate(image_ROIs):
        for ind_img, img in enumerate(images):
            img = img[ROI[1]:(ROI[3] + 1), ROI[0]:(ROI[2] + 1), :]
            img = np.flip(img, axis=0)
            img = np.flip(img, axis=1)

            print("image size = ", str(img.shape))
            plt.figure()
            plt.imshow(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # plt.figure()
            # plt.imshow(thresh_inv,cmap=plt.cm.gray)
            # find contours
            contours = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  #
            boxes = [cv2.boundingRect(c) for c in contours]
            boxes = [x for x in boxes if x[2] > img.shape[1] * 0.75 and x[3] > img.shape[0] * 0.75]
            if len(boxes) == 0:
                print("No valid boxes found!")
                continue
            boxes = [x for x in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)]
            box = boxes[0]
            img = img[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2]), :]

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            plt.title("image %i, ROI %i" % (ind_img + 1, ind_roi + 1))

            dx = int(np.round(img.shape[1] / ROI[-1]))
            for part in range(ROI[-1]):
                sub_img = img[:, (dx * part):(dx * (part + 1)), :]

                plt.subplot(2, ROI[-1], ROI[-1] + part + 1)
                plt.imshow(sub_img)

                # count+=1
                # plt.imsave(OUTPUT + "%i.png" % count,sub_img)
                ALL_images.append(sub_img)

    import random

    random.seed(1)
    random.shuffle(ALL_images)

    for k, img in enumerate(ALL_images[0:400]):
        plt.imsave(OUTPUT + "%i.png" % (k + 1), img)

elif MODE == 1:

    save_file += "_type%i" % MODEL_TYPE

    X = [cv2.imread(TRAIN_DATA[0] + os.sep + '%i.png' % (k)) for k in range(1,400)]+[cv2.imread(TRAIN_DATA[1] + os.sep + '%i.png' % (k)) for k in range(1,398)]
    X = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in X]
    Y = np.concatenate([np.array(pd.read_csv(TRAIN_DATA[0] + os.sep + "labels.csv",header=None)),np.array(pd.read_csv(TRAIN_DATA[1] + os.sep + "labels.csv",header=None))]).flatten()

    X = [resize_image(x, size=IMG_SIZE,PAD = PAD) for x in X]
    X = np.stack(X,axis=0)

    np.random.seed(666)
    null = np.where(Y==-1)[0]
    np.random.shuffle(null)
    idx = list(np.random.permutation(X.shape[0],))
    idx = [x for x in idx if x not in null]
    idx = idx + [x for x in null[0:80]]
    np.random.shuffle(idx)

    X = np.take(X,idx,axis=0)
    Y = np.take(Y,idx,axis=0)

    assert np.min(Y)>-2 and np.max(Y)<10
    assert np.max(X)<=255 and np.min(X)>=0

    N_training = int(X.shape[0]*0.80)
    N_test = X.shape[0] - N_training
    X_train = X[0:N_training,:,:]
    X_test = X[N_training:,:,:]
    Y_train = Y[0:N_training,]
    Y_test = Y[N_training:,]

    # PART 2 - model training
    if len(X.shape)==3:
        X_train = np.expand_dims(X_train,3)
        X_test = np.expand_dims(X_test,3)

    # DATA FORMAT (samples,rows,cols,channels)
    print("PART 2 - training model")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from keras import backend as K

    from sklearn.metrics import confusion_matrix
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    from keras_buoy.models import ResumableModel
    from keras.utils.np_utils import to_categorical
    import tensorflow_model_optimization as tfmot

    N_targets = len(np.unique(Y))
    Y_train = to_categorical(Y_train,N_targets) # here -1 goes to index 9
    Y_test = to_categorical(Y_test,N_targets)

    epochs = 5000
    batch_size = 64
    filtersize = 32
    input_shape=(IMG_SIZE[0],IMG_SIZE[1],NB_CHANNELS)

    if MODEL_TYPE == 1:
        model = Sequential()

        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation="relu"))

        model.add(keras.layers.Dense(10, activation="softmax"))

    elif MODEL_TYPE == 2:

        model = Sequential()
        model.add(keras.layers.Conv2D(filtersize, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filtersize*2, kernel_size=(3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.30))
        model.add( keras.layers.Dense(10, activation="softmax"))

    elif MODEL_TYPE == 3:

        cnn = Sequential()
        cnn.add(keras.layers.Conv2D(filters=filtersize,
                       kernel_size=(3,3),
                       strides=(1, 1),
                       padding='same',
                       input_shape=input_shape))
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                             strides=2))
        cnn.add(keras.layers.Conv2D(filters=filtersize*2,
                       kernel_size=(3,3),
                       strides=(1, 1),
                       padding='valid'))
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                             strides=2))
        cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(64))
        cnn.add(keras.layers.Activation('relu'))
        cnn.add(keras.layers.Dropout(0.25))
        cnn.add(keras.layers.Dense(N_targets))
        cnn.add(keras.layers.Activation("softmax"))

        model = cnn

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.build(input_shape = [None] + list(input_shape))

    model.summary()

    best_acc = 0
    if LOAD_OLD_MODEL:
        model = keras.models.load_model(save_file)
    if 1:
        # With data augmentation to prevent overfitting
        datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = [0.95,1.20], # Randomly zoom image
                shear_range=0.10,
                width_shift_range=0.08,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.08,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        datagen.fit(X_train)
        # fits the model on batches with real-time data augmentation:
        # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
        #                    steps_per_epoch=len(x_train) / 32,
        #                    epochs=epochs)
        print("Starting training")
        N_samples = X_train.shape[0]
        print_interval = int(N_samples/10.025)
        train_scores = []
        plt.figure()

        start = time.time()
        if 0:
            for e in range(epochs):
                print(' starting epoch', e+1)
                print(' ...sample ',end='')
                samples = 0
                print_samples = 0
                for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size,shuffle=True,seed = e):
                    model.fit(x_batch, y_batch,verbose=0)
                    samples += batch_size
                    print_samples += batch_size
                    if print_samples>=print_interval:
                        print(' %i' % samples,end='')
                        print_samples = 0
                    if samples >= 2*N_samples:
                        # we need to break the loop by hand because
                        # the generator loops indefinitely
                        break
                print(' finished!')
                print(' evaluating...')
                score = model.evaluate(test_datagen.flow(X_test,Y_test,batch_size=200),verbose=0,batch_size=200)
                train_scores.append(score)
                print('Test score:', score[0])
                print('Test accuracy:', score[1])

                if score[1]>best_acc:
                    print("better model, saving new snapshot")
                    model.save_weights(save_file)
                    best_acc = score[1]

                plt.clf()
                plt.plot([x[1] for x in train_scores])
                plt.title("Training epoch %i" % (e+1))
                plt.xlabel("Epoch")
                plt.ylabel("Test accuracy")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.01)
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,restore_best_weights=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_file,
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size,shuffle=True), verbose=1,epochs=10) # warm-up
            history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size,shuffle=True), verbose=1,epochs=epochs,validation_data=test_datagen.flow(X_test,Y_test,batch_size=200),callbacks=[early_stopping,model_checkpoint])

        end = time.time()
        print('Training time (minutes):', (end - start) / 60)

        print("PART 3 - plotting results")

        # Fit the model
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig("val_accuracy.png")

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("val_loss.png")

    X_test = X_test.astype(np.float32)/255
    y_pred = model.predict(X_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # Plot Confusion matrix
    mat = confusion_matrix(np.argmax(Y_test, 1), np.argmax(y_pred,1)) # Confusion matrix
    sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.show(block=False)
    plt.savefig("confusion_matrix.png")

    fig, axis = plt.subplots(4, 4, figsize=(10, 12))
    for i, ax in enumerate(axis.flat):
        ax.imshow(np.squeeze(X_test[i,:,:,:]))
        ax.set(title = f"real={Y_test[i,:].argmax()}, pred={y_pred[i,:].argmax()}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("predictions_matrix.png")

    '''
    img1 = process_image(cv2.imread('test1.png'),mean_X)
    img2 = process_image(cv2.imread('test2.png'),mean_X)
    
    img1_pred = model.predict(img1).argmax()
    print("img1 real 6, predicted %i" % img1_pred)
    
    img2_pred = model.predict(img2).argmax()
    print("img2 real 4, predicted %i" % img2_pred)
    '''

elif MODE == 2:

    from keras.models import load_model

    video_file = r'D:\JanneK\Documents\git_repos\MyMediaPlayer-main\mydata\WIN_20210713_20_33_27_Pro.mp4'
    video_ROI = (470, 191, 599, 258, 3)

    save_file += "_type%i" % MODEL_TYPE

    model = load_model(save_file)

    NAME = 'Detector'
    cv2.namedWindow(NAME)
    capture = cv2.VideoCapture(video_file)
    prev_boxes=[]
    while True:
        if capture.isOpened():
            # Read frame
            (status, orig_frame) = capture.read()
            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            frame_digits,prev_boxes,rectangles = image_preprocessor(frame,video_ROI,prev_boxes=prev_boxes)
            if len(frame_digits)!=video_ROI[-1]:
                print("Failed to find individual digits (%i found)!" % len(frame_digits))
                continue
            frame_digits = [resize_image(x,IMG_SIZE) for x in frame_digits]
            frame_digits = np.stack(frame_digits,axis=0).astype(np.float32)/255.0
            digits = model.predict(frame_digits)
            digits = np.argmax(digits, 1)
            digits[digits==10]=-1

            if cv2.getWindowProperty(NAME, 0) >= 0:
                for rect,digit in zip(rectangles,digits):
                    orig_frame = cv2.rectangle(orig_frame,(rect[0],rect[2]),(rect[1],rect[3]),(0, 255, 0), 2)
                orig_frame = cv2.flip(orig_frame, 0)
                orig_frame = cv2.flip(orig_frame, 1)
                for rect,digit in zip(rectangles,digits):
                    text_location = [orig_frame.shape[1]-rect[1],orig_frame.shape[0]-rect[3]-4]
                    orig_frame = cv2.putText(orig_frame,str(digit) if digit>=0 else "",text_location, cv2.FONT_HERSHEY_SIMPLEX,1, [0,255,0], 2, cv2.LINE_AA)
                cv2.imshow(NAME,orig_frame)

            print("reading = %s" % str(digits))
            key = cv2.waitKey(10)
        else:
            print("Capture terminated!")

