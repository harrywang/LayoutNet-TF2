import tensorflow as tf
import config


class Dataset:
    def __init__(self):
        # prepare dataset
        dataset = tf.data.TFRecordDataset(config.filenamequeue)
        dataset = dataset.map(self._decode_tfrecords)

        # TODO: change the buffer_size
        dataset = dataset.shuffle(buffer_size=1024,
                                  reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=config.batch_size)
        dataset = dataset.as_numpy_iterator()

        self.dataset = dataset

    # decode function for dataset
    def _decode_tfrecords(self, example_string):
        features = tf.io.parse_single_example(
            example_string,
            features={
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

    def next(self):
        return self.dataset.next()
