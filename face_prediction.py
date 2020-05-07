import cv2


class GenderDetector:
    def __init__(self):
        # to detect face in image
        self.face_cascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')

        # to detect gender of face
        self.gender_net = cv2.dnn.readNetFromCaffe('weigths/face/gender.prototxt', 'weigths/face/gender.caffemodel')
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.gender_list = ['Male', 'Female']

    def detect_face(self, image):
        img = cv2.imread(image)
        faces_list = []
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = image.copy()[y:y + h, x:x + w]
            faces_list.append(face)
        return faces_list

    def detect_gender(self, person_image):
        face_list = self.detect_face(person_image)
        if face_list:
            # input conversion
            blob = cv2.dnn.blobFromImage(face_list[0], 1, (227, 227), self.mean_values, swapRB=False)
            # Gender detection on face
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            # converting class from confidences
            gender = self.gender_list[gender_preds[0].argmax()]
            return gender_preds[0]
        else:
            return None
