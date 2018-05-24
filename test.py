from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import csv as csv

CHECK_POINT_PATH = 'check_point/train_model.ckpt-2'
NUM_KAGGLE_TEST = 12500
BATCH_SIZE = 50
batch_test_set = data_processing.pre_processing(data_set='test', batch_size=BATCH_SIZE)

prediction_file = open('kaggle_result_file.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(['id', 'label'])

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)
    logits, _ = nets.vgg.vgg_19(inputs=images, num_classes=2, dropout_keep_prob=keep_prob, is_training=False)
    variables_to_restore = slim.get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)

    pros = tf.nn.softmax(logits)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, CHECK_POINT_PATH)

        for i in range(NUM_KAGGLE_TEST // BATCH_SIZE):
            batch_pros = sess.run(pros, feed_dict={images: batch_test_set['images'][i], keep_prob: 1.0})
            print('Batch %d of %d'%(i, NUM_KAGGLE_TEST // BATCH_SIZE))
            for j in range(BATCH_SIZE):
                temp = 0.0
                if batch_pros[j, 1] > 0.5:
                   temp = 0.995
                else:
                   temp = 0.005                 
                prediction_file_object.writerow([ i * BATCH_SIZE + j + 1, temp])
prediction_file.close()
