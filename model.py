import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class EmbeddingSemvec(keras.Model):
    def __init__(self):
        super(EmbeddingSemvec, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.cat_fc = Dense(48,
                            activation=activation,
                            kernel_initializer=initializer,
                            use_bias=False)
        self.txt_fc = Dense(48,
                            activation=activation,
                            kernel_initializer=initializer,
                            use_bias=False)
        self.img_fc = Dense(48,
                            activation=activation,
                            kernel_initializer=initializer,
                            use_bias=False)

        self.fc = Dense(32,
                        activation=activation,
                        kernel_initializer=initializer,
                        use_bias=False)

    def call(self, inputs, is_training=None):
        category = tf.concat([
            inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6],
            inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6], inputs[:, 0:6],
            inputs[:, 0:6], inputs[:, 0:6]
        ], 1)
        textratio = tf.concat([
            inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13],
            inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13], inputs[:, 6:13],
            inputs[:, 6:13], inputs[:, 6:13]
        ], 1)
        imgratio = tf.concat([
            inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23],
            inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23],
            inputs[:, 13:23], inputs[:, 13:23], inputs[:, 13:23], inputs[:,
                                                                         13:23]
        ], 1)

        cat = self.cat_fc(category)
        txt = self.txt_fc(textratio)
        img = self.img_fc(imgratio)

        x = tf.concat([cat, txt, img], 1)

        return self.fc(x)


class EmbeddingImg(keras.Model):
    def __init__(self):
        super(EmbeddingImg, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.img_fc1 = Dense(512,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.img_fc2 = Dense(256,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.img_fc3 = Dense(128,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)

    def call(self, inputs, is_training=None):
        x = tf.reduce_mean(inputs, [1, 2])
        x = self.img_fc1(x)
        x = self.img_fc2(x)
        x = self.img_fc3(x)

        return x


class EmbeddingTxt(keras.Model):
    def __init__(self):
        super(EmbeddingTxt, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.txt_fc1 = Dense(256,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.txt_fc2 = Dense(256,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.txt_fc3 = Dense(128,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)

    def call(self, inputs, is_training=None):
        x = self.txt_fc1(inputs)
        x = self.txt_fc2(x)
        x = self.txt_fc3(x)

        return x


class EmbeddingFusion(keras.Model):
    def __init__(self):
        super(EmbeddingFusion, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        self.fusion_fc1 = Dense(256,
                                activation=activation,
                                kernel_initializer=initializer,
                                use_bias=False)
        self.fusion_fc2 = Dense(128,
                                activation=activation,
                                kernel_initializer=initializer,
                                use_bias=False)

    def call(self, input1, input2, input3, is_training=None):
        inputs = tf.concat([input1, input2, input3], 1)
        x = self.fusion_fc1(inputs)
        x = self.fusion_fc2(x)

        return x


class Gen(keras.Model):
    def __init__(self):
        super(Gen, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.projection = Dense(4 * 4 * 512,
                                activation=activation,
                                kernel_initializer=initializer,
                                use_bias=False)
        self.reshape = Reshape((4, 4, 512))
        self.bn_0 = BatchNormalization()

        self.conv_tp1 = Conv2DTranspose(256,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation,
                                        kernel_initializer=initializer,
                                        use_bias=False)
        self.bn_1 = BatchNormalization()
        self.conv_tp2 = Conv2DTranspose(128,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation,
                                        kernel_initializer=initializer,
                                        use_bias=False)
        self.bn_2 = BatchNormalization()
        self.conv_tp3 = Conv2DTranspose(64,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=activation,
                                        kernel_initializer=initializer,
                                        use_bias=False)
        self.bn_3 = BatchNormalization()
        self.conv_tp4 = Conv2DTranspose(3,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=initializer,
                                        use_bias=False)

    def call(self, z, is_training, y=None):
        if y is not None:
            inputs = tf.concat((z, y), 1)
        else:
            inputs = z
        x = self.projection(inputs)
        x = self.reshape(x)
        x = self.bn_0(x, is_training)

        x = self.conv_tp1(x)
        x = self.bn_1(x, is_training)

        x = self.conv_tp2(x)
        x = self.bn_2(x, is_training)

        x = self.conv_tp3(x)
        x = self.bn_3(x, is_training)

        x = self.conv_tp4(x)

        return x


class Disc(keras.Model):
    def __init__(self):
        super(Disc, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.leaky_relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.conv_0 = Conv2D(64,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)

        self.conv_1 = Conv2D(128,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(256,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_2 = BatchNormalization()

        self.conv_3 = Conv2D(512,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_3 = BatchNormalization()

        self.conv_4 = Conv2D(128,
                             kernel_size=4,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=False)

        self.fc = Dense(1,
                        activation=tf.nn.tanh,
                        kernel_initializer=initializer,
                        use_bias=False)

    def call(self, inputs, is_training, y, z):
        if y is not None:
            x = tf.concat([inputs, y], 3)
        else:
            x = inputs

        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.bn_1(x, is_training)

        x = self.conv_2(x)
        x = self.bn_2(x, is_training)

        x = self.conv_3(x)
        x = self.bn_3(x, is_training)

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
        super(Encoder, self).__init__()
        initializer = tf.keras.initializers.truncated_normal(stddev=0.02)
        activation = tf.nn.leaky_relu
        kernel_size = 5
        strides = (2, 2)
        padding = 'same'

        self.conv_0 = Conv2D(64,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)

        self.conv_1 = Conv2D(128,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_1 = BatchNormalization()

        self.conv_2 = Conv2D(256,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_2 = BatchNormalization()

        self.conv_3 = Conv2D(512,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             activation=activation,
                             kernel_initializer=initializer,
                             use_bias=False)
        self.bn_3 = BatchNormalization()

        self.conv_4_1 = Conv2D(128,
                               kernel_size=4,
                               strides=(1, 1),
                               padding='valid',
                               activation=None,
                               use_bias=False)
        self.conv_4_2 = Conv2D(128,
                               kernel_size=4,
                               strides=(1, 1),
                               padding='valid',
                               activation=None,
                               use_bias=False)

    def call(self, inputs, is_training, y=None):
        x = self.conv_0(inputs)

        x = self.conv_1(x)
        x = self.bn_1(x, is_training)

        x = self.conv_2(x)
        x = self.bn_2(x, is_training)

        x = self.conv_3(x)
        x = self.bn_3(x, is_training)

        if y is not None:
            x = tf.concat([x, y], 3)

        x_1 = self.conv_4_1(x)
        x_2 = self.conv_4_2(x)

        x_1 = tf.squeeze(x_1, [1, 2])
        x_2 = tf.squeeze(x_2, [1, 2])

        return x_1, x_2


class LayoutNet(keras.Model):
    def __init__(self, config):
        super(LayoutNet, self).__init__()
        self.embeddingSemvec = EmbeddingSemvec()
        self.embeddingImg = EmbeddingImg()
        self.embeddingTxt = EmbeddingTxt()

        self.embeddingFusion = EmbeddingFusion()

        self.encoder = Encoder()
        self.generator = Gen()
        self.discriminator = Disc()

        self.config = config

    def call(self, x, y, tr, ir, img, tex, z, is_training=True):
        config = self.config

        category = tf.one_hot(y, depth=config.y_dim)
        textratio = tf.one_hot(tr, depth=config.tr_dim)
        imgratio = tf.one_hot(ir, depth=config.ir_dim)
        x_labeltmp = tf.concat([category, textratio, imgratio], 1)

        var_label = self.embeddingSemvec(x_labeltmp, is_training)
        img_fea = self.embeddingImg(img, is_training)
        tex_fea = self.embeddingTxt(tex, is_training)

        y_label = self.embeddingFusion(var_label, img_fea, tex_fea,
                                       is_training)

        ydis_label = tf.reshape(
            y_label, shape=(-1, 1, 1, config.latent_dim)) * tf.ones(
                [config.batch_size, 64, 64, config.latent_dim])

        encoderdis_label = tf.reshape(
            y_label, shape=(-1, 1, 1, config.latent_dim)) * tf.ones(
                [config.batch_size, 4, 4, config.latent_dim])

        randomz = tf.random.normal([config.batch_size, config.z_dim])

        z_mean, z_log_sigma_sq = self.encoder(x,
                                              is_training,
                                              y=encoderdis_label)
        E = z_mean + tf.exp(z_log_sigma_sq) * randomz

        G = self.generator(z, is_training, y=y_label)
        G_recon = self.generator(E, is_training, y=y_label)

        D_real = self.discriminator(x, is_training, y=ydis_label, z=z)
        D_fake = self.discriminator(G, is_training, y=ydis_label, z=z)

        return z_mean, z_log_sigma_sq, E, G, G_recon, D_real, D_fake
