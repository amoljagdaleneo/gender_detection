import tensorflow.keras as keras
import glob
import cv2
import create_embedding
import numpy as np


class TrainClassifier:
    def __init__(self):
        self.keras_model = keras
        self.male = 0
        self.female = 1

    def create_model(self):
        model = self.keras_model.models.Sequential([
            self.keras_model.layers.Dense(5, bias_initializer='zeros', kernel_initializer='glorot_uniform',
                                          input_shape=(128,),
                                          activation="relu"),
            self.keras_model.layers.Dense(25, activation="relu"),
            self.keras_model.layers.Dense(20, activation="softmax")
        ])
        model.compile(loss=self.keras_model.losses.categorical_crossentropy,
                      optimizer=self.keras_model.optimizers.Adam(lr=0.01))
        return model

    def dataset_parser(self, dataset_path):
        train_x = []
        train_y = []

        # initalize embedding model
        self.emb_model = create_embedding.InitModel()
        self.emb_model.init_embeding()

        male_datapath = dataset_path + "/male"
        # read images and get embedding for 0-male and 1-female
        for image in glob.glob(male_datapath + "/*jpg"):
            image_cv = cv2.imread(image)
            image_emb = self.emb_model.human_vector(image_cv)
            train_x.append(image_emb)
            output = self.keras_model.utils.to_categorical(self.male, num_classes=2)
            train_y.append(output)

        # for female dataset
        female_datapath = dataset_path + "/female"
        for image in glob.glob(female_datapath + "/*jpg"):
            image_cv = cv2.imread(image)
            image_emb = self.emb_model.human_vector(image_cv)
            output = self.keras_model.utils.to_categorical(self.female, num_classes=2)
            train_x.append(image_emb)
            train_y.append(output)
        return train_y, train_y

    def train_model(self, dataset_path):
        #  data gathering
        train_x, train_y = self.dataset_parser(dataset_path)
        inputs = np.array(train_x).reshape(len(train_x), 128)
        output = np.array(train_y).reshape(len(train_y), 2)

        # creating model
        model = self.create_model()
        model.fit(inputs, output, verbose=0)
        model.save("gender_classifier.h5")
