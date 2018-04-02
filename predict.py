import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os
import winsound
import time


def __init__():
    global term_crit
    global track_window
    r, h, c, w = 75, 300, 150, 300
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1000)
    track_window = (c, r, w, h)


def preprocess(image):
    open_cv_image = np.array(image)
    blur = cv2.GaussianBlur(open_cv_image, (3, 3), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, processed_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    im_pil = Image.fromarray(processed_image)
    th3 = np.array(im_pil)
    return th3


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


def returnSegmented(img):
    global term_crit
    # Replace with the image you want as your comparison
    roi = cv2.imread('Hand Crop2.jpg')

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


def playSound(letter):
    # The text that you want to convert to audio
    mytext = letter
    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.write_to_fp(f)
    winsound.PlaySound(f, winsound.SND_FILENAME)
    return


__init__()
font = cv2.FONT_HERSHEY_SIMPLEX
topRight = (10, 100)
fontScale = 0.9
fontColor = (255, 255, 255)
lineType = 2

tracking_type = "Dynamic"
model = load_model('26_aug_kag+cust.h5')

img_height = 28
img_width = 28

y = 100
x = 100
h = 200
w = 200

textStartingY = 100

letter_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

cap = cv2.VideoCapture(1)

f = TemporaryFile()
old_time = time.time()
current_time = time.time()

while (True):

    # Capture frame-by-frame
    cat, frame = cap.read(1)

    if tracking_type is "Static":
        frameResized = frame[y:y + h, x:x + w]

        frameResized = cv2.resize(frameResized, (img_height, img_width))
        # Our operations on the frame come here
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Converting to black white like the training images
        frameResized = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)

        # reshaping for nn
        frameResized = frameResized.reshape(1, img_height, img_width, 1)

    if tracking_type is "Dynamic":
        img, crop = returnSegmented(frame)
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        crop = cv2.resize(crop, (img_width, img_height))

        cv2.imshow('Crop', crop)

        frameResized = crop.reshape(1, img_height, img_width, 1)

    result = model.predict(frameResized, batch_size=1)[0]  # Predict

    guessed_letter = letter_map[np.argmax(result)]
    for i in range(0, len(result)):
        strToPrint0 = str(letter_map[i]) + ': ' + str(round(result[i], 2))
        if (i < 13):  # Adds second row of 12 characters starting back at the top
            cv2.putText(frame, strToPrint0, (10, textStartingY + i * 30), font, fontScale, fontColor, lineType)
        else:
            cv2.putText(frame, strToPrint0, (300, textStartingY + (i - 12) * 30), font, fontScale, fontColor, lineType)
    cv2.putText(frame, guessed_letter, (220, textStartingY), font, fontScale, fontColor, lineType)

    if time.time() - old_time > 5:
        old_time = time.time()
        print ("it's been a minute")
    cv2.imshow('frame', frame)

    playSound(guessed_letter)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

