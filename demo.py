from typing import Text
import tensorflow as tf
from model import *
import numpy as np
from preprocessing import *
import config
from PIL import Image
import matplotlib.pyplot as plt


class LayoutNetDemo:
    def __init__(self, checkpoint_path):
        # define model
        self.layoutnet = LayoutNet(config)

        # restore from latest checkpoint
        self.layoutnet.load_weights(checkpoint_path)

        # visual feature extract
        self.vis_fea = VisualFeatureExtract()

        # text feature extract
        self.txt_fea = TextFeatureExtract()

        # category, text_ratio and image ratio handler
        self.sem_vec = AttributeFeatureHandler()

    def generate(self, category, text_ratio, image_ratio, image_path,
                 keywords_list, z):
        # process user input
        y, tr, ir = self.sem_vec.get(category=category,
                                     text_ratio=text_ratio,
                                     image_ratio=image_ratio)

        # extract image feature
        img_feature = self.vis_fea.extract(image_path)

        # extract text feature according to keywords
        txt_feature = self.txt_fea.extract(keywords_list)

        # generate result
        generated = self.layoutnet.generate(y, tr, ir, img_feature,
                                            txt_feature, z)
        generated = (generated + 1.) / 2.
        image = generated[0]

        # return raw to postprocessing
        # image = Image.fromarray(np.uint8(image * 255))

        return image


if __name__ == '__main__':
    demo = LayoutNetDemo(checkpoint_path='./checkpoints/ckpt-300')

    category = 'food'
    text_ratio = 0.5
    image_ratio = 0.5
    image_path = ['./test1.jpg', './test2.jpg']
    keywords_list = ['Taste', 'wine', 'restaurant', 'fruit', 'market']

    # generate random latent variable
    z = np.random.normal(0.0, 1.0, size=(1, config.z_dim)).astype(np.float32)

    # generate result
    image = demo.generate(category, text_ratio, image_ratio, image_path,
                          keywords_list, z)
    image.save('./demo.png')