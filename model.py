import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class EmbeddingSemvec(keras.Model):
    def __init__(self):
        super(EmbeddingSemvec, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.cat_fc = Dense(48, activation=activation,
                            kernel_initializer=initializer)
        self.txt_fc = Dense(48, activation=activation,
                            kernel_initializer=initializer)
        self.img_fc = Dense(48, activation=activation,
                            kernel_initializer=initializer)

        self.fc = Dense(32, activation=activation,
                        kernel_initializer=initializer)

    def call(self, inputs, training=None):
        category = tf.concat([inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6],
                              inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6]], 1)
        textratio = tf.concat([inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13],
                               inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13]], 1)
        imgratio = tf.concat([inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23],
                              inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23]], 1)

        cat = self.cat_fc(category)
        txt = self.txt_fc(textratio)
        img = self.img_fc(imgratio)

        x = tf.concat([cat, txt, img], 1)

        return self.net(x)


class EmbeddingImg(keras.Model):
    def __init__(self):
        super(EmbeddingImg, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.img_fc1 = Dense(512, activation=activation,
                             kernel_initializer=initializer)
        self.img_fc2 = Dense(256, activation=activation,
                             kernel_initializer=initializer)
        self.img_fc3 = Dense(128, activation=activation,
                             kernel_initializer=initializer)

    def call(self, inputs, training=None):
        x = tf.reduce_mean(inputs, [1, 2])
        x = self.img_fc1(x)
        x = self.img_fc2(x)
        x = self.img_fc3(x)

        return x


class EmbeddingTxt(keras.Model):
    def __init__(self):
        super(EmbeddingTxt).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.txt_fc1 = Dense(256, activation=activation,
                             kernel_initializer=initializer)
        self.txt_fc2 = Dense(256, activation=activation,
                             kernel_initializer=initializer)
        self.txt_fc3 = Dense(128, activation=activation,
                             kernel_initializer=initializer)

    def call(self, inputs, training=None):
        x = self.txt_fc1(inputs)
        x = self.txt_fc2(x)
        x = self.txt_fc3(x)

        return x


class EmbeddingFusion(keras.Model):
    def __init__(self):
        super(EmbeddingFusion).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.fusion_fc1 = Dense(
            256, activation=activation, kernel_initializer=initializer)
        self.fusion_fc2 = Dense(
            128, activation=activation, kernel_initializer=initializer)

    def call(self, inputs, training=None):
        x = self.fusion_fc1(inputs)
        x = self.fusion_fc2(x)

        return x


class Gen(keras.Model):
    def __init__(self):
        super(Gen).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.projection = Dense(
            4 * 4 * 512, activation=activation, kernel_initializer=initializer)
        self.reshape = Reshape(4 * 4 * 512)
        self.bn = BatchNormalization()

        self.conv_tp1 = Conv2DTranspose(
            256, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=initializer)
        self.conv_tp2 = Conv2DTranspose(
            128, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=initializer)
        self.conv_tp3 = Conv2DTranspose(
            64, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=initializer)
        self.conv_tp4 = Conv2DTranspose(
            3, kernel_size=kernel_size, strides=strides, padding=padding, activation=tf.nn.tanh, kernel_initializer=initializer)

    def call(self, z, is_training, y=None):
        inputs = tf.concat((z, y), 1) if y else z
        x = self.projection(inputs)
        x = self.reshape(x)
        x = self.bn(x)

        x = self.conv_tp1(x)
        x = self.conv_tp2(x)
        x = self.conv_tp3(x)
        x = self.conv_tp4(x)

        return x


class Disc(keras.Model):
    def __init__(self):
        super(Disc).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.leaky_relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.conv_0 = Conv2D(64, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)

        self.conv_1 = Conv2D(128, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(256, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_2 = BatchNormalization()

        self.conv_3 = Conv2D(512, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_3 = BatchNormalization()

        self.conv_4 = Conv2D(128, kernel_size=4,
                             strides=(1, 1), padding='valid')

        self.fc = Dense(1, activation=tf.nn.tanh,
                        kernel_initializer=initializer)

    def call(self, inputs, is_training, y, z):
        x = tf.concat([inputs, y], 3) if y else inputs

        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)

        x = self.conv_4(x)

        if z is None:
            x = tf.squeeze(x, [1, 2])
        else:
            x = tf.squeeze(x, [1, 2])
            x = tf.concat((x, z), 1)
            x = self.fc(x)

        return x


class Encoder(keras.Model):
    def __init__(self):
        super(Encoder).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.leaky_relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.conv_0 = Conv2D(64, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)

        self.conv_1 = Conv2D(128, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(256, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_2 = BatchNormalization()

        self.conv_3 = Conv2D(512, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation=activation, kernel_initializer=initializer)
        self.bn_3 = BatchNormalization()

        self.conv_4_1 = Conv2D(128, kernel_size=4, strides=(
            1, 1), padding='valid', activation=None)
        self.conv_4_2 = Conv2D(128, kernel_size=4, strides=(
            1, 1), padding='valid', activation=None)

    def call(self, inputs, is_training, y=None):
        x = self.conv_0(inputs)

        x = self.conv_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)

        x = tf.concat([x, y], 3) if x else x

        x_1 = self.conv_4_1(x)
        x_2 = self.conv_4_2(x)

        return x_1, x_2
