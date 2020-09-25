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

        return image


if __name__ == '__main__':
    print('loading the checkpoint...')
    demo = LayoutNetDemo(checkpoint_path='./checkpoints/ckpt-300')

    category = 'food'
    text_ratio = 0.5
    image_ratio = 0.5
    image_path1 = ['./demo/food.jpg', './demo/wine.jpg']
    image_path2 = ['./demo/fashion.jpg']  # not food related
    keywords_list = ['Taste', 'wine', 'restaurant', 'fruit', 'market']

    # generate random latent variable
    z = np.random.normal(0.0, 1.0, size=(1, config.z_dim)).astype(np.float32)

    # generate result
    print('generating demo 1')
<<<<<<< HEAD
    image = demo.generate(category, text_ratio, image_ratio, image_path1,
                          keywords_list, z)
=======
    image_raw = demo.generate(category, text_ratio, image_ratio, image_path1,
                              keywords_list, z)

    image = Image.fromarray(np.uint8(image_raw * 255))
>>>>>>> 47cdcde05df520c338fb899663494fb4c82ec0ce
    image.save('./demo/demo1.png')

    # generate result
    print('generating demo 2')
<<<<<<< HEAD
    image = demo.generate(category, text_ratio, image_ratio, image_path2,
                          keywords_list, z)
    image.save('./demp/demo2.png')
    
=======
    image_raw = demo.generate(category, text_ratio, image_ratio, image_path2,
                              keywords_list, z)
    image = Image.fromarray(np.uint8(image_raw * 255))
    image.save('./demo/demo2.png')
>>>>>>> 47cdcde05df520c338fb899663494fb4c82ec0ce
