import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import datetime as time
import os
import colorSeg


def streamProcessed():
    # intitalize the color segmentation module
    colorSeg.__init__()
    iterator = 0
    numImages = 0
    record = True
    # load the image
    #Check if error given on run
    cap = cv2.VideoCapture(0)
    calibrated = False
    # these lines of code are for a video for demonstrating the process
    # img = cv2.imread(args["image"])
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('testvid.avi', fourcc, 20.0, (640,480))
    while (cap.isOpened):
        _, img = cap.read()
        # img = img[::2, ::2]
        # img = img.copy()
        # img = cv2.imread(args["image"])

        # hand is calibrated to the environment
        if calibrated:
            _, img = cap.read()
            cv2.imshow("Tracking hand", img)
            cv2.createTrackbar("Window Size", "Tracking hand", colorSeg.track_window[2], 400, colorSeg.updateRectangle);
            processedimg, croppedImg = colorSeg.returnSegmented(img)
            # cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
            overlay = img.copy()
            aslLetterIndex = 1
            cv2.putText(overlay, "Letter: " + alphabet[aslLetterIndex],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.addWeighted(overlay, .2, img, .8, 0, img)
            cv2.imshow("Tracking hand", overlay)
        else:
            while(True):
                _, img = cap.read()
                k = cv2.waitKey(30) & 0xff
                # finish the calibration process by pressing the space bar
                if k == 32:
                    cv2.destroyWindow
                    break
                # draws the blue rectangle, change the tuple values to change the size of the rectangle
                cv2.rectangle(img, (200, 200), (230, 230), 255, 2)
                # copy the image before the text is applied
                overlay = img.copy()
                cv2.putText(overlay, "Please place the middle of your hand in the blue box and press the space bar to calibrate",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
                cv2.addWeighted(overlay, .2, img, .8, 0, img)
                cv2.imshow("countdown",overlay)
            calibrate(img[200:230,200:230])
        calibrated = True
        k = cv2.waitKey(30) & 0xff
        # breaks with esc
        if k == 27:
            break
        iterator += 1
    cap.release()
    #out.release()
    cv2.destroyAllWindows


def calibrate(img):
    colorSeg.calibration(img)

if __name__ == "__main__":

    alphabet = []
    for letter in range(65, 91):
        alphabet.append(chr(letter))
    streamProcessed()
