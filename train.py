from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import os
import os.path
import time
TRAIN_LOG_DIR = os.path.join('Log/train/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = 'check_point/train_model.ckpt'
VALIDATION_LOG_DIR = 'Log/validation/'
VGG_19_MODEL_DIR = 'check_point/vgg_19.ckpt'
BATCH_SIZE = 32
EPOCH = 3
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
    images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, len(data_processing.IMG_CLASSES)])
    keep_prob = tf.placeholder(tf.float32)
    with slim.arg_scope(nets.vgg.vgg_arg_scope()):
        logits, _ = nets.vgg.vgg_19(inputs=images, num_classes=2, dropout_keep_prob=keep_prob, is_training=True)
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_19/fc8'])
    restorer = tf.train.Saver(variables_to_restore)
    
    with tf.name_scope('cross_entropy'):
         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.summary.scalar('cross_entropy', loss)
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
         accuracy = get_accuracy(logits, labels)
    tf.summary.scalar('accuracy', accuracy)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(TRAIN_LOG_DIR)
        #train_writer.add_summary(sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, VGG_19_MODEL_DIR)
        step = 0
        for ep in range(EPOCH):
            all_accuracy = 0
            all_loss = 0
            for i in range(images_num['train'] // BATCH_SIZE):
                _, accuracy_out, loss_out, summary = sess.run([optimizer, accuracy, loss, merged],
                                       feed_dict={images: batch_train_set['images'][i], \
                                                  labels: batch_train_set['labels'][i],
                                                  keep_prob: 0.5})
                train_writer.add_summary(summary, step)
                step += 1
                all_accuracy += accuracy_out
                all_loss += loss_out
                if i % 10 == 0:
                    print("Epoch %d: Batch %d accuracy is %.2f; Batch loss is %.5f" %(ep + 1, i, accuracy_out, loss_out))
                    

            print("Epoch %d: Train accuracy is %.2f; Train loss is %.5f" %(ep + 1, all_accuracy / (images_num['train'] // BATCH_SIZE), all_loss / (images_num['train'] // BATCH_SIZE)))
            
            all_accuracy = 0
            all_loss = 0
            for i in range(images_num['validation'] // BATCH_SIZE):
                accuracy_out, loss_out = sess.run([accuracy, loss],
                                       feed_dict={images: batch_validation_set['images'][i], \
                                                  labels: batch_validation_set['labels'][i], keep_prob: 1.0})
                all_accuracy += accuracy_out
                all_loss += loss_out

            print("Epoch %d: Validation accuracy is %.2f; Validation loss is %.5f" %(ep + 1, all_accuracy / (images_num['validation'] // BATCH_SIZE), all_loss / (images_num['validation'] // BATCH_SIZE)))
            
            saver.save(sess, TRAIN_CHECK_POINT, global_step=ep)
