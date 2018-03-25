from imutils.video import VideoStream
from PIL import Image
import os
import time
import argparse
import sys

def collect_specific_hand(character_wanted, name):
    vs = VideoStream(-1).start() #start up the video stream
    if os.path.exists("hand/%s" % character_wanted): #if you don't have a file for the character you specified, it'll be created
        pass
    else:
        os.makedirs("hand/%s" % character_wanted)
    print("Starting data collection for character", character_wanted, "in 5 seconds, get ready...")
    time.sleep(5)
    print("Collecting now!")
    num_of_images = 50
    while True and num_of_images != 0:
        num_of_images = num_of_images - 1
        frame = vs.read()
        im = Image.fromarray(frame)
        im.save("hand/%s/%s.png" % (character_wanted, name + str(num_of_images))) #save it into created file
    print("Done!")
    return True

def collect_not_hand(name):
    vs = VideoStream(-1).start()
    print("Starting data collection for non-hands in 5 seconds, get ready...")
    time.sleep(5)
    print("Collecting now!")
    num_of_images = 200
    while True and num_of_images != 0:
        num_of_images = num_of_images - 1
        frame = vs.read()
        im = Image.fromarray(frame)
        im.save("not_hand/%s.png" % (name + str(num_of_images)))
    print("Done!")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # get both arguements
    ap.add_argument("-c", "--character", required=True,
                    help="Character Wanted, type 'None' if you're making negative images")
    ap.add_argument("-n", "--name", required=True,
                    help="Your name, used to save file.")
    args = vars(ap.parse_args())
    if args["character"] == 'None':
        collect_not_hand(args["name"].lower())
    else:
        collect_specific_hand(args["character"].lower(), args["name"].lower())
    sys.exit(0)
