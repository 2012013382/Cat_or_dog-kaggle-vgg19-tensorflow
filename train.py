from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

TRAIN_LOG_DIR = 'Log/train/'
TRAIN_CHECK_POINT = 'check_point/train_model.ckpt'
VALIDATION_LOG_DIR = 'Log/validation/'
VGG_16_MODEL_DIR = 'check_point/vgg_16.ckpt'
BATCH_SIZE = 32
EPOCH = 10
if not tf.gfile.Exists(TRAIN_LOG_DIR):
    tf.gfile.MakeDirs(TRAIN_LOG_DIR)

if not tf.gfile.Exists(VALIDATION_LOG_DIR):
    tf.gfile.MakeDirs(VALIDATION_LOG_DIR)

batch_train_set, batch_validation_set, images_num = data_processing.pre_processing(data_set='train', batch_size=BATCH_SIZE)

def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

with tf.Graph().as_default():
    train_images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    train_labels = tf.placeholder(tf.float32, [BATCH_SIZE, len(data_processing.IMG_CLASSES)])
    logits, _ = nets.vgg.vgg_16(inputs=train_images, num_classes=2, is_training=True)
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
    restorer = tf.train.Saver(variables_to_restore)

    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels))
    validation_loss = train_loss
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)

    train_accuracy = get_accuracy(logits, train_labels)
    validation_accuracy = train_accuracy
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, VGG_16_MODEL_DIR)

        for ep in range(EPOCH):
            accuracy = 0
            loss = 0
            for i in range(images_num['train'] // BATCH_SIZE):
                _, accuracy_out, loss_out = sess.run([optimizer, train_accuracy, train_loss],
                                       feed_dict={train_images: batch_train_set['images'][i], \
                                                  train_labels: batch_train_set['labels'][i]})
                accuracy += accuracy_out
                loss += loss_out
                if i % 10 == 0:
                    print("Epoch %d: Batch %d accuracy is %.2f; Batch loss is %.5f" %(ep + 1, i, accuracy_out, loss_out))

            print("Epoch %d: Train accuracy is %.2f; Train loss is %.5f" %(ep + 1, accuracy / (images_num['train'] // BATCH_SIZE), loss / (images_num['train'] // BATCH_SIZE)))
            accuracy = 0
            loss = 0
            for i in range(images_num['validation'] // BATCH_SIZE):
                accuracy_out, loss_out = sess.run([validation_accuracy, validation_loss],
                                       feed_dict={train_images: batch_validation_set['images'][i], \
                                                  train_labels: batch_validation_set['labels'][i]})
                accuracy += accuracy_out
                loss += loss_out

            print("Epoch %d: Validation accuracy is %.2f; Validation loss is %.5f" %(ep + 1, accuracy / (images_num['validation'] // BATCH_SIZE), loss / (images_num['validation'] // BATCH_SIZE)))
            saver.save(sess, TRAIN_CHECK_POINT, global_step=ep)
