import cv2
import pymongo
from person_identifier import PersonDetector
from face_prediction import GenderDetector
from body_gender_predict import PredictGender
import time
import numpy as np
import sys


class DetectAndUpload:
    def __init__(self):
        # database connectivity
        cursor = pymongo.MongoClient("mongodb://localhost:27017/")
        gender_db = cursor["person_db"]
        self.gender_result = gender_db["person_gender"]

        # initialising modules
        self.person_detector = PersonDetector()
        self.face_gender = GenderDetector()
        self.body_gender = PredictGender()

    # main function run
    def detect_and_save(self, image):
        detected_persons = self.person_detector.detect(image)
        frame = cv2.imread(image)
        if frame is None:
            print("please insert valid image")
        for detected_person in detected_persons:
            face_result = self.face_gender.detect_gender(detected_persons)
            body_result = self.body_gender.predict_image(detected_person)

            # No face is detected
            if face_result is None:
                # face is not detected
                print(body_result)
                if int(np.argmax(body_result)):
                    result = "Male"
                else:
                    result = "Female"
                self.save(detected_person, result, image)
                return result

            # Result of both body and face are same
            elif int(np.argmax(body_result)) == int(np.argmax(face_result)):
                if int(np.argmax(face_result)):
                    result = "Male"
                else:
                    result = "Female"
                self.save(detected_person, result, image)
                return result
            else:
                # face and body result combined
                gender = ["Male", "Female"]
                face_result = [i * 0.7 for i in face_result]
                body_result = [i * 0.3 for i in body_result]

                final_result = np.array([sum(i) for i in zip(face_result, body_result)]).argmax()
                result = gender[final_result]
                self.save(detected_person, result, image)
                return result

    # writing result in database
    def create_db_document(self, result):
        self.gender_result.insert_one(result)

    # write crop image and make db writable result
    def save(self, detected_person, result, image):
        cropped_image_path = "predicted_images/" + result + str(time.time()) + ".jpg"
        cv2.imwrite(cropped_image_path, detected_person)
        result_to_write = {"Original": image, "Cropped": cropped_image_path, "Gender": result}
        self.create_db_document(result_to_write)


if __name__ == "__main__":
    if sys.argv[1] == "-imagepath":
        obj = DetectAndUpload()
        obj.detect_and_save(sys.argv[2])
    else:
        print("use -imagepath to give image")
