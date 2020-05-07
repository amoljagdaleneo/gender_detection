# Gender Detection

Gender detection using both facial features and full body features

# Requirements:

- face detection using opencv haar feature xml <a href="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml">Link</a>
- gender detection from face data (model) <a href="">Link</a>
- person detection models <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">Link</a>
- trained model of train_gender_classifier.py
- mongoDB running 

Run 

    detect_and_upload.py -image_path= path/to/image


Output will be saved in mongo database

For training model run 
    
    python train_gender_classifier.py
    
This will train on images in the dataset folder. The images in the dataset are cropped images of persons detected