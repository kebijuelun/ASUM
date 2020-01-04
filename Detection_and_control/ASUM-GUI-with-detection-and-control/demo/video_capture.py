from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

print("[INFO] starting threaded video stream...")
stream = WebcamVideoStream(src=0).start()  # default camera

# read video for detection
# stream = cv2.VideoCapture("/home/lwz/data/videos/slices7_cut.avi")
# stream = cv2.VideoCapture("/home/lwz/data/videos/slices7_cut.avi")
frame_width = int(1024)
frame_height = int(768)

ori_out = cv2.VideoWriter('/media/lwz/BCF60E29F60DE50C/test-2019-0321.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 35, (frame_width, frame_height))

time.sleep(1.0)
# start fps timer
# loop over frames from the video file stream
while True:


    start_time = time.time()

    # grab next frame
    # ret, frame = stream.read()
    frame = stream.read()
    # ori_out.write(frame)
    # if ret%2 != 0:
    #     continue

    key = cv2.waitKey(1) & 0xFF

    # update FPS counter
    # fps.update()
    ori_out.write(frame)

    # det_frame = predict(frame)

    # det_out.write(det_frame)
    end_time = time.time()
    current_fps = 1/(end_time - start_time)
    print("[INFO] current. FPS: {:.2f}".format(current_fps))
    # show current FPS on screen

    # cv2.putText(frame, "FPS: {:.2f}".format(current_fps), (10, 30), 1, 2,
    #             (125, 100, 150), 2)

    # keybindings for display
    if key == ord('p'):  # pause
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord('p'):  # resume
                break
    cv2.imshow('frame', frame)
    if key == 27:  # exit
        stream.stop()
        break