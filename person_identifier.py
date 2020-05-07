import cv2

model_name = "rcnn-inception"
threshold = 0.80


class PersonDetector:
    def __init__(self):
        # initialize network with model weights
        self.tensorflow_net = cv2.dnn.readNetFromTensorflow('weights/' + model_name + '/frozen_inference_graph.pb',
                                                            'weights/' + model_name + '/graph.pbtxt')

    # detects persons in a image and return list of cropped persons
    def detect(self, image_name):
        image = cv2.imread(image_name)
        self.tensorflow_net.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
        network_output = self.tensorflow_net.forward()
        rows, cols, channels = image.shape
        detected_person = []
        for detection in network_output[0, 0]:
            score = float(detection[2])
            if score > threshold and detection[1] == 0.0:  # threshold for accurate detections
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                croped_image = image[top:bottom, left:right]
                detected_person.append(croped_image)
        return detected_person
