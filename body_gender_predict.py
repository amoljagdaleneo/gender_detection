import cv2
import create_embedding
import train_gender_classifier


class PredictGender:
    def __init__(self):
        self.emb_model = create_embedding.InitModel()
        self.emb_model.init_embeding()
        self.classifier = train_gender_classifier.TrainClassifier().create_model()

    def predict_image(self, image):
        image_cv = cv2.imread(image)
        image_emb = self.emb_model.human_vector(image_cv)
        predictions = self.classifier.predict([image_emb])
        return predictions
