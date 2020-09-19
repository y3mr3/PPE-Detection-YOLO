



# Importing needed libraries
import numpy as np
import cv2
import time


# Defining function for processing given image
def yolo3(path):

    

    # Reading image with OpenCV library
    # In this way image is opened already as numpy array
    # WARNING! OpenCV by default reads images in BGR format
    image_BGR = cv2.imread(path)

    # Check point
    # Showing image shape
    print()
    print('Image shape:', image_BGR.shape)  # tuple of (466, 700, 3)

    # Getting spatial dimension of input image
    h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

    # Check point
    # Showing height an width of image
    print('Image height={0} and width={1}'.format(h, w))  # 466 700

    

    # Getting blob from input image
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
    # from input image after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of images, number of channels, width and height
    # E.G.:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    # Check point
    print('Blob shape:', blob.shape)  # (1, 3, 416, 416)


    # Loading COCO class labels from file
    # Opening file


    with open('yolo/class.names') as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet('yolo/yolov3_ppe_test.cfg',
                                         'yolo/yolov3_ppe_train_15000.weights')

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

    # Check point
    # print()
    # print(layers_names_all)

    # Getting only output layers' names that we need from YOLO v3 algorithm
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    # Check point
    # print()
    # print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

    # Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5

    # Setting threshold for filtering weak bounding boxes
    # with non-maximum suppression
    threshold = 0.3

    # Generating colours for representing every detected object
    # with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Check point
    # print()
    # print(type(colours))  # <class 'numpy.ndarray'>
    # print(colours.shape)  # (80, 3)
    # print(colours[0])  # [172  10 127]

   

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Showing spent time for forward pass
    print()
    print('Objects Detection took {:.5f} seconds'.format(end - start))

  

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial image size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original image and in this way get coordinates for center
                # of bounding box, its width and height for original image
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)


    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Defining counter for detected objects
    counter = 1

    # Checking if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

            # Incrementing counter
            counter += 1

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

            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    # Comparing how many objects where before non-maximum suppression
    # and left after
    print()
    print('Total objects been detected:', len(bounding_boxes))
    print('Number of objects left after non-maximum suppression:', counter - 1)

  

    # Saving resulted image in jpg format by OpenCV function
    # that uses extension to choose format to save with
    cv2.imwrite('result.jpg', image_BGR)
