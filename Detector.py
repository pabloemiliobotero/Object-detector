import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker



######################################

from centroidtracker import CentroidTracker
from threading import Thread
import time
import numpy as np

frame_rate_calc = 1
freq = cv2.getTickFrequency()
resW, resH = "480x640".split('x')
imW, imH = int(resW), int(resH)
leftcount = 0
rightcount = 0
obsFrames = 0

height, width = int(resW), int(resH)

mitad = imH/2

videostream =[]
# Path to label map file
PATH_TO_LABELS = os.path.join('.', 'model_data', 'coco_classes.txt')

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = [np.subtract(dict2[key][0], dict1[key][0]),dict2[key][1]]
   return dict3
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(480,640),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects = {}
old_objects = {}
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

model = YOLO("yolov8n.pt")
rects = []
labelFinal = []
centroids = []
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
#cap = cv2.VideoCapture(os.path.join('.', 'data', 'people.mp4'))
#ret, frame1 = cap.read()
i=0
while True:
    print(i)
    i += 1

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # On the next loop set the value of these objects as old for comparison
    old_objects.update(objects)

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_resized = cv2.resize(frame_rgb, (width, height))
    # img_yuv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
    # clahe = cv2.createCLAHE()
    # img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    # # convert the YUV image back to RGB format
    # frame_resized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # input_data = np.expand_dims(frame_resized, axis=0)

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if ((score > 0.5)):
                object_name = labels[int(class_id)]  # Look up object name from "labels" array using class index
                box = np.array([x1, y1, x2, y2])
                rects.append(box.astype("int"))
                detections.append([x1, y1, x2, y2, score])
                # Draw label
                label = '%s: %d%%' % (object_name, int(score * 100))  # Example: 'person: 72%'
                cv2.putText(frame, str(label), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)
                labelFinal.append(label)
                #centroid = (int((int(x2) + int(x1)) / 2), int((int(y2) + int(y1)) / 2))

        try:
            objects = ct.update(rects, labelFinal)
            tracker.update(frame, detections)
        except:
            cv2.putText(frame, 'Any objects detected', (240,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print("any objects detected")
        #centroids.append(centroid)
        #objects = tracker.update(rects, labelFinal)

    # calculate the difference between this and the previous frame
    x = DictDiff(objects, old_objects)

    if (len(tracker.tracks) > 0):
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            centroid = (int((int(x2)+int(x1))/2), int((int(y2)+int(y1))/2))
            cv2.circle(frame, centroid, 5, (255, 0, 255), 3)
            #cv2.putText(frame, str(track_id), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2, cv2.LINE_AA)
#########################

    # Draw framerate in corner of frame
    #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    #print("FPS: " + str(frame_rate_calc))
    # All the results have been drawn on the frame, so it's time to display it.

    # cv2.namedWindow("Object detector", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Object detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Object detector', frame)
    cv2.waitKey(25)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    #count number of frames for direction calculation

cv2.destroyAllWindows()
videostream.stop()
######################################
# video_path = os.path.join('.', 'data', 'people.mp4')
# video_out_path = os.path.join('.', 'out.mp4')
#
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))
#
# model = YOLO("yolov8n.pt")
#
# tracker = Tracker()
#
# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
#
# detection_threshold = 0.5
# while True:
#     t1 = cv2.getTickCount()
#     results = model(frame)
#
#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             if ((class_id == 0 ) and (score > detection_threshold)):
#                 detections.append([x1, y1, x2, y2, score])
#         #print(result)
#         tracker.update(frame, detections)
#
#         for track in tracker.tracks:
#             bbox = track.bbox
#             x1, y1, x2, y2 = bbox
#             track_id = track.track_id
#
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
#             centroid = (int((int(x2)+int(x1))/2), int((int(y2)+int(y1))/2))
#             cv2.circle(frame, centroid, 5, (255, 0, 255), 3)
#             cv2.putText(frame, str(track_id), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#             cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2, cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#     cap_out.write(frame)
#     ret, frame = cap.read()
#     cv2.waitKey(25)
#
#     t2 = cv2.getTickCount()
#     time1 = (t2 - t1) / freq
#     frame_rate_calc = 1 / time1
#
# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()