import os
import scipy.misc
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from model import *
import config


# decode function for dataset
def _decode_tfrecords(example_string):
    features = tf.io.parse_single_example(example_string, features={
        "label": tf.io.FixedLenFeature([], tf.int64),
        "textRatio": tf.io.FixedLenFeature([], tf.int64),
        "imgRatio": tf.io.FixedLenFeature([], tf.int64),
        'visualfea': tf.io.FixedLenFeature([], tf.string),
        'textualfea': tf.io.FixedLenFeature([], tf.string),
        "img_raw": tf.io.FixedLenFeature([], tf.string)
    })

    image = tf.io.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [60, 45, 3])
    image = tf.cast(image, tf.float32)

    resized_image = tf.image.resize_with_crop_or_pad(image, 64, 64)
    resized_image = resized_image / 127.5 - 1.

    label = tf.cast(features['label'], tf.int32)

    textRatio = tf.cast(features['textRatio'], tf.int32)
    imgRatio = tf.cast(features['imgRatio'], tf.int32)

    visualfea = tf.io.decode_raw(features['visualfea'], tf.float32)
    visualfea = tf.reshape(visualfea, [14, 14, 512])

    textualfea = tf.io.decode_raw(features['textualfea'], tf.float32)
    textualfea = tf.reshape(textualfea, [300])

    return resized_image, label, textRatio, imgRatio, visualfea, textualfea


# prepare dataset
dataset = tf.data.TFRecordDataset(config.filenamequeue)
dataset.map(_decode_tfrecords)
# TODO: change the buffer_size
dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
dataset.batch(batch_size=config.batch_size)

# create model
layoutnet = LayoutNet(config)

# define loss and opt

def discriminator_loss(D_real, D_fake):
    loss_D_real = tf.reduce_mean(tf.nn.l2_loss(D_real - tf.ones_like(D_real)))
    loss_D_fake = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.zeros_like(D_fake)))

    loss_D = loss_D_real + loss_D_fake

    return loss_D


def generator_loss(x, z_log_sigma_sq, z_mean, D_fake, G_recon):
    loss_Gls = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.ones_like(D_fake)))

    kl_div = -0.5 * tf.reduce_sum(1 + 2 * z_log_sigma_sq -
                                  tf.square(z_mean) - tf.exp(2 * z_log_sigma_sq), 1)

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshpae(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    loss_G3 = tf.reduce_mean(kl_div + recon_loss) / 64 / 64 / 3
    # compared to encoder_loss
    # generator_loss don't consider kl_div anymore
    # why?
    recon_loss = tf.reduce_mean(recon_loss) / 64 / 64 /3
    
    loss_G = loss_Gls + recon_loss

    return loss_G


def encoder_loss(x, z_log_sigma_sq, z_mean, G_recon):
    # TODO: the dimension is hardcoded, it's not good
    kl_div = -0.5 * tf.reduce_sum(1 + 2 * z_log_sigma_sq -
                                  tf.square(z_mean) - tf.exp(2 * z_log_sigma_sq), 1)

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshpae(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    loss_E = tf.reduce_sum(kl_div, recon_loss) / 64 / 64 / 3

    return loss_E


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1).minimize
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1)

# save checkpoints

checkpoint_dir = config.checkpoint_dir
checkpoint_prefix = config.checkpoint_basename
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    encoder_optimizer=encoder_optimizer,
    generator=layoutnet.generator,
    discriminator=layoutnet.discriminator,
    encoder=layoutnet.encoder)


# define the training loop

num_examples_to_generate = 16
# from TF2.3 DCGAN documents:
## We will reuse this seed overtime (so it's easier)
## to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, config.z_dim])


@tf.function
def train_step():
    pass
