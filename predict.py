import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

def __init__():
    global term_crit
    global track_window
    r, h, c, w = 75, 300, 150, 300
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1000)
    track_window = (c, r, w, h)

def meanshift(dst, img):
    global track_window
    deltaTrack = track_window
    ret, track_window2 = cv2.meanShift(dst, track_window, term_crit)
    # print(np.sqrt([(deltaTrack[0]-track_window[0])**2+(deltaTrack[2]-track_window[2])**2]))
    # x,y,w,h = track_window2
    x, y = track_window2[0], track_window2[1]
    w, h = track_window[2], track_window[3]
    track_window = x, y, w, h
    img2 = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    croppedImg = img[y:y + h, x:x + w]
    return img2, croppedImg

def calibration(cap):
    while(True):
        _,img = cap.read()
        k = cv2.waitKey(30) & 0xff
        # finish the calibration process by pressing the space bar
        if k == 32:
            cv2.destroyWindow
            break
        # draws the blue rectangle, change the tuple values to change the size of the rectangle
        cv2.rectangle(img, (200, 200), (230, 230), 255, 2)
        # copy the image before the text is applied
        overlay = img.copy()
        cv2.putText(overlay, "Please place the middle of your hand in the blue box",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay,
                    "and press the space bar to calibrate.",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # cv2.createTrackbar("Window Size", "Tracking hand", track_window[2], 400, updateRectangle);
        cv2.addWeighted(overlay, .2, img, .8, 0, img)
        cv2.imshow("countdown",overlay)
    return img[200:230,200:230]
def returnSegmented(img,calibrated_roi):
    global term_crit
    # Replace with the image you want as your comparison
    roi = calibrated_roi

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    target = img
    hsvt = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)

    # calculating object histogram
    # roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)

    dst = cv2.calcBackProject([hsvt], [0], roi_hist, [0, 180], 1)
    img2, croppedImg = meanshift(dst, img)
    return img2, croppedImg


def playSound(guessed_letter):
    playsound("sound/" + str(guessed_letter) + ".mp3")
    return

__init__()
font = cv2.FONT_HERSHEY_SIMPLEX
topRight = (10, 100)
fontScale = 2
fontColor = (255, 255, 255)
lineType = 2

model = load_model('model.h5')

img_height = 28
img_width = 28

y = 100
x = 100
h = 200
w = 200

textStartingY = 100

letter_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

cap = cv2.VideoCapture(0)

# finds the initital region of interest
roi = calibration(cap)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('testvid.avi', fourcc, 20.0, (640,480))

while (True):

    # Capture frame-by-frame
    cat, frame = cap.read(1)

    img, crop = returnSegmented(frame,roi)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    crop = cv2.resize(crop, (img_width, img_height))


    frameResized = crop.reshape(1, img_height, img_width, 1)

    result = model.predict(frameResized, batch_size=1)[0]  # Predict

    guessed_letter = letter_map[np.argmax(result)]
    # for i in range(0, len(result)):
    #     strToPrint0 = str(letter_map[i]) + ': ' + str(round(result[i], 2))
    #     if (i < 13):  # Adds second row of 12 characters starting back at the top
    #         cv2.putText(frame, strToPrint0, (10, textStartingY + i * 30), font, fontScale, fontColor, lineType)
    #     else:
    #         cv2.putText(frame, strToPrint0, (300, textStartingY + (i - 12) * 30), font, fontScale, fontColor, lineType)
    cv2.putText(frame, guessed_letter, (20, 60), font, fontScale, fontColor, lineType)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# out.release()
cap.release()
cv2.destroyAllWindows()
