from typing import Text
import tensorflow as tf
from model import *
import numpy as np
from preprocessing import *
import config
from PIL import Image
import matplotlib.pyplot as plt

# define model
layoutnet = LayoutNet(config)

# restore from latest checkpoint
layoutnet.load_weights('./checkpoints/ckpt-300')

# visual feature extract
vis_fea = VisualFeatureExtract()

# text feature extract
txt_fea = TextFeatureExtract()

# category, text_ratio and image ratio handler
sem_vec = AttributeFeatureHandler()

if __name__ == '__main__':
    # process user input

    y, tr, ir = sem_vec.get(category='food', text_ratio=0.5, image_ratio=0.5)

    print('Extracting Image Feature...')
    img_feature = vis_fea.extract('./test.jpg')

    print('Extracting Text Feature...')
    txt_feature = txt_fea.extract(
        ['Taste', 'wine', 'restaurant', 'fruit', 'market'])

    # generate random latent variable
    z = np.random.normal(0.0, 1.0, size=(1, config.z_dim)).astype(np.float32)

    # generate result
    generated = layoutnet.generate(y, tr, ir, img_feature, txt_feature, z)
    generated = (generated + 1.) / 2.

    image = generated[0]

    image = Image.fromarray(np.uint8(image * 255))
    image.save('./demo.png')