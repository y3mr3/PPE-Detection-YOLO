


# Importing needed libraries
import numpy as np
import cv2
import time


"""
Start of:
Reading stream video from camera
"""

# Defining 'VideoCapture' object
# and reading stream video from camera
camera = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)

def make_1080p():
    camera.set(3, 1920)
    camera.set(4, 1080)

def make_720p():
    camera.set(3, 1280)
    camera.set(4, 720)

def make_480p():
    camera.set(3, 640)
    camera.set(4, 480)

def change_res(width, height):
    camera.set(3, width)
    camera.set(4, height)

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading stream video from camera
"""


"""
Start of:
Loading YOLO v3 network
"""

with open('yolo/class.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV

network = cv2.dnn.readNetFromDarknet('yolo/yolov3_ppe_test.cfg',
                                     'yolo/yolov3_ppe_train_9000.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.7

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]

"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""

fpsLimit = 1
startTime = time.time()

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame from 
 
    _, frame = camera.read()
    
    nowTime = time.time()
    if (int(nowTime - startTime)) > fpsLimit:
        if w is None or h is None:
        # Slicing from tuple only first two elements
            h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        bounding_boxes = []
        confidences = []
        class_numbers = []
        
        for result in output_from_network:
                for detected_objects in result:
                    scores = detected_objects[5:]
                    class_current = np.argmax(scores)
                    confidence_current = scores[class_current]
                    if confidence_current > probability_minimum:
                            box_current = detected_objects[0:4] * np.array([w, h, w, h])
                            x_center, y_center, box_width, box_height = box_current
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))
                            bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)
                        
        
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
                               
        if len(results) > 0:
        # Going through indexes of results
            for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
    
            # Preparing colour for current bounding box
            # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()

            # # # Check point
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127]

            # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                        
        cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO v3 Real Time Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        startTime = time.time() # reset time
 
# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()


"""
Some comments

cv2.VideoCapture(0)

To capture video, it is needed to create VideoCapture object.
Its argument can be camera's index or name of video file.
Camera index is usually 0 for built-in one.
Try to select other cameras by passing 1, 2, 3, etc.
"""
