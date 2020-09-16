import tensorflow as tf
from tensorflow.python.types.core import Value
from model import *
from tensorflow.keras.preprocessing import image
import numpy as np
import gensim.downloader as api


class VisualFeatureExtract:
    def __init__(self):
        self.vgg16_base = tf.keras.applications.VGG16(include_top=False,
                                         input_shape=(224, 224, 3))
        self.vgg16_feature_extract = tf.keras.Model(
            inputs=self.vgg16_base.input,
            outputs=self.vgg16_base.get_layer('block5_conv3').output)

    def extract(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg16.preprocess_input(img)  # it's important to preprocess

        result = self.vgg16_feature_extract(img)
        # result = np.squeeze(result)

        return result


class TextFeatureExtract:
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')

    def extract(self, word_list):
        result = []
        for item in word_list:
            result.append(np.array(self.model[item]))
        result = np.array(result)
        result = result.sum(axis=0)
        result = np.expand_dims(result, axis=0)

        return result

    def word_fea_extract(self, word):
        return self.model[word]


class AttributeFeatureHandler:
    # category:     int64 0-5
    # text ratio:   0.1 - 0.7  7 scale -> int 0-6
    # image ratio:  0.1 - 1.0 10 scale -> int 0-9
    def __init__(self):
        self.category_list = ['fashion', 'food', 'news', 'science', 'travel', 'wedding']

    def get(self, category, text_ratio, image_ratio):
        assert type(category) == str
        assert type(text_ratio) == float and text_ratio >= 0
        assert type(image_ratio) == float and image_ratio >= 0
        
        # embedding category using integer
        category = category.lower()
        if category not in self.category_list:
            assert ValueError('%s is not a valid category' % category)

        cate_embedding = self.category_list.index(category)
        
        tr_embedding = min(round(text_ratio * 10), 6)
        ir_embedding = min(round(image_ratio * 10), 9)
        
        return cate_embedding, tr_embedding, ir_embedding



if __name__ == '__main__':
    # vis fea example
    # visfea = VisualFeatureExtract()
    # print(visfea.vis_fea_extract('./test.jpg').shape)

    # txt fea example
    txtfea = TextFeatureExtract()

    result = txtfea.txt_fea_extract(['cat', 'dog', 'plane'])
    print(result)
    print(result.size)
