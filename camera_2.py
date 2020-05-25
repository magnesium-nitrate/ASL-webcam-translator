import cv2
import imutils
import numpy as np
import keras
import time
from keras.models import load_model
model = load_model('CNNmodel.h5')
import matplotlib.pyplot as plt

bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.GaussianBlur(thresholded,(5,5),0)
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    #top, right, bottom, left = 10, 350, 225, 590
    top, right, bottom, left = 100, 450, 225, 575

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]
        #print(frame[100][100])
        # get the ROI
        roi = frame[top:bottom, right:left]
        #print(roi.shape)
        im = roi#.flatten()
        print(im.shape)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        print(im.shape)
        im = cv2.resize(im,(28,28),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        print(im.shape)
        #im = im[:,:,0]
        im = im.reshape(1,28,28,1)
        #plt.imshow(im[0,:,:,0])
        #plt.show()
        #print(im)
        #time.sleep(3)
        #im = np.flip(im)
        pred_probab = model.predict(im)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        print(chr(ord('@')+pred_class+1))
        #cv2.putText(roi,chr(ord('@')+pred_class+1),(700,300),cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #print(pred_class)
        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (255,0,0), 2)
        cv2.putText(clone,chr(ord('@')+pred_class+1),(100,400),cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
