import tensorflow as tf
from weights.reid.nets import resnet_v1_50 as model
import weights.reid.heads.fc1024_normalize as head
import cv2
import numpy as np


class InitModel:
    def __init__(self):
        self.session_v1 = tf.compat.v1.Session()
        self.images = tf.zeros([1, 256, 128, 3], dtype=tf.float32)

    def init_embeding(self):
        endpoints, body_prefix = model.endpoints(self.images, is_training=False)
        print(endpoints, body_prefix)
        with tf.name_scope('head'):
            self.endpoints = head.head(endpoints, 128, is_training=False)
        tf.compat.v1.train.Saver().restore(self.session_v1, 'models/reid/model/checkpoint-25000')

    # embedding created in this function
    def human_vector(self, img):
        resize_img = cv2.resize(img, (128, 256))
        resize_img = np.expand_dims(resize_img, axis=0)
        emb = self.session_v1.run(self.endpoints['emb'], feed_dict={self.images: resize_img})
        return emb

    def __del__(self):
        self.session_v1.close()
