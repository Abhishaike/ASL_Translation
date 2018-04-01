import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import datetime as time
import os
import colorSeg


def streamProcessed():
    colorSeg.__init__()
    iterator = 0
    numImages = 0
    record = True
    # load the image
    #Check if error given on run
    cap = cv2.VideoCapture(0)
    calibrated = True
    _, img = cap.read()
    # img = cv2.imread(args["image"])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('testvid.avi', fourcc, 20.0, (640,480))
    cv2.imshow("Tracking hand", img)
    cv2.createTrackbar("Window Size", "Tracking hand", colorSeg.track_window[2], 400, colorSeg.updateRectangle);
    while (cap.isOpened):
        # start timer
        _, img = cap.read()

        # img = img[::2, ::2]
        # img = img.copy()
        # img = cv2.imread(args["image"])
        if calibrated:
            processedimg, croppedImg = colorSeg.returnSegmented(img)
            cv2.imshow("Tracking hand", processedimg)
            # cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
            overlay = img.copy()
            aslLetterIndex = 1
            cv2.putText(overlay, "Letter: " + alphabet[aslLetterIndex],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.addWeighted(overlay, .2, img, .8, 0, img)
        else:
            cv2.rectangle(img, (200, 200), (230, 230), 255, 2)
            cv2.imshow("Tracking hand", img)
            overlay = img.copy()
            cv2.putText(overlay, "Please place a section of your hand in the blue box and click the button",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
            cv2.addWeighted(overlay, .2, img, .8, 0, img)
            cv2.createButton("Calibrate", calibrate(img))
            cv2.imshow(img)
        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break
        iterator += 1
        # stop the clock
        # print(end - start)
        # close the image window when a key is pressed
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # quit
    cap.release()
    out.release()
    cv2.destroyAllWindows


def calibrate(img):
    colorSeg.calibration(img)

if __name__ == "__main__":

    alphabet = []
    for letter in range(65, 91):
        alphabet.append(chr(letter))
    streamProcessed()