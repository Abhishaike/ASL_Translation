import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import datetime as time
import os

global track_window


def __init__():
    global term_crit
    global track_window
    roi_img = 
    r, h, c, w = 75, 300, 150, 300
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1000)
    track_window = (c, r, w, h)

def calibration(img):
    roi_img = img


def returnSegmented(img):
    global term_crit
    global roi_img
    #Replace with the image you want as your comparison
    roi = roi_img
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    target = img;
    hsvt = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)

    # calculating object histogram
    # roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)

    dst = cv2.calcBackProject([hsvt], [0], roi_hist, [0, 180], 1)
    img2, croppedImg = meanshift(dst, img)
    return img2, croppedImg

def meanshift(dst, img):
    print(str(term_crit))
    global track_window
    deltaTrack = track_window
    ret, track_window2 = cv2.meanShift(dst, track_window, term_crit)
    #print(np.sqrt([(deltaTrack[0]-track_window[0])**2+(deltaTrack[2]-track_window[2])**2]))
    #x,y,w,h = track_window2
    x, y = track_window2[0], track_window2[1]
    w, h = track_window[2], track_window[3]
    print(w,h)
    track_window = x,y,w,h
    img2 = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    croppedImg = img[y:y+h, x:x+w]
    return img2, croppedImg


def camshift(dst, img):
    global track_window
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
       # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(img,[pts],True, 255,2)
    return img2


def streamSegmented():
    iterator = 0
    numImages = 0
    global term_crit
    global track_window
    global noargs
    global character
    global name
    record = True
    # load the image
    #Check if error given on run
    cap = cv2.VideoCapture(0)

    _, img = cap.read()
    # img = cv2.imread(args["image"])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('testvid.avi', fourcc, 20.0, (640,480))
    cv2.imshow("Tracking hand", img)
    cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
    while (cap.isOpened):
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 200)
        # start timer
        _, img = cap.read()
        #img = img[::2, ::2]
        #img = img.copy()
        # img = cv2.imread(args["image"])
        processedimg, croppedImg = returnSegmented(img)
        cv2.imshow("Tracking hand", processedimg)
        #cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);

        k = cv2.waitKey(30) & 0xff
        if(record == True):
           out.write(processedimg)
        if (iterator %10 == 0 and noargs == False):
            if(len(croppedImg) > 100):
                croppedImg = cv2.resize(croppedImg,(100, 100))
            cv2.imwrite("/home/evan/Documents/ACM/aslwebserver/pictures/%s/%s" % (character, name+str(int(iterator/10))), croppedImg)

        elif (noargs == True and k==32):
            if(len(croppedImg) > 100):
                croppedImg = cv2.resize(croppedImg,(100, 100))
            cv2.imwrite("/home/evan/Documents/ACM/aslwebserver/pictures/unlabelled/%s/img%s.png" % (date, numImages), croppedImg)
            numImages += 1
        if k == 27:
            break
        iterator +=1
        # stop the clock
        # print(end - start)
        # close the image window when a key is pressed
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # quit
    cap.release()
    out.release()
    cv2.destroyAllWindows

def updateRectangle(size):
    print(size)
    global track_window
    track_window = (track_window[0], track_window[1], size, size)
    print(track_window)

if __name__ == "__main__":
    global character
    global name
    global noargs
    global date
    now = time.datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    noargs = False
    ap = argparse.ArgumentParser()
    # get both arguements
    ap.add_argument("-c", "--character", required=False,
                    help="Character Wanted, type 'None' if you're making negative images")
    ap.add_argument("-n", "--name", required=False,
                    help="Your name, used to save file.")
    args = vars(ap.parse_args())
    if(args["character"] != None and args["name"] != None):
        character = args["character"].lower()
        name = args["name"].lower()
        if os.path.exists(
                "pictures/%s" % character):  # if you don't have a file for the character you specified, it'll be created
            pass
        else:
            os.makedirs("pictures/%s" % character)
    else:
        noargs = True
        os.makedirs("pictures/unlabelled/%s" % str(date))
    __init__()
    streamSegmented()