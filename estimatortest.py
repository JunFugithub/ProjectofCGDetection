import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parser(record):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) # [self.patch_size, self.patch_size])
    label = tf.cast(parsed['label'], tf.int32)
    return image,label
def importdata(filename, epoch):

    dataset  = tf.data.TFRecordDataset(filename).shuffle(80)
    dataset  = dataset.map(parser)
    dataset  = dataset.repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    img,lbl      = iterator.get_next()
    return tf.reshape(img, [-1,650,650,1]),lbl #tf.reshape(tf.cast(tf.one_hot(lbl, 2), tf.int32), [-1,2]) #tf.reshape(lbl, [-1,1])

def model_fn(features, labels, mode):


    kernel = np.array([[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]])
    kernal_5x5_3 = kernel.reshape([5, 5, 1, 1])
    x = tf.nn.conv2d(input=features, filter=kernal_5x5_3, padding='VALID', strides=[1, 2, 2, 1], name='HPF')
    # x = tf.layers.conv2d(inputs=kernel, filters=1,kernel_size=[5,5],padding='valid',kernel_initializer=kernel)
    # x = tf.layers.average_pooling2d(inputs=features, pool_size=[2, 2], strides=2, padding='valid', name='HPF',)

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    conv1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[5, 5], padding='same', name='CONV1', kernel_regularizer=regularizer)
    conv1_bn = tf.layers.batch_normalization(inputs=conv1, axis=-1, momentum=0.9, name="CONV1_BN", training=mode == tf.estimator.ModeKeys.TRAIN)
    conv1_pl = tf.layers.average_pooling2d(inputs=conv1_bn, padding='same', pool_size=[2, 2], name='CONV1_PL',
                                           strides=2)

    conv2 = tf.layers.conv2d(inputs=conv1_pl, filters=16, kernel_size=[5, 5], padding='same', name='CONV2', kernel_regularizer=regularizer)
    conv2_bn = tf.layers.batch_normalization(inputs=conv2, axis=-1, momentum=0.9, name="CONV2_BN",
                                             training=mode == tf.estimator.ModeKeys.TRAIN)
    conv2_pl = tf.layers.average_pooling2d(inputs=conv2_bn, padding='same', pool_size=[2, 2], name='CONV2_PL',
                                           strides=2)

    conv3 = tf.layers.conv2d(inputs=conv2_pl, filters=32, kernel_size=[3, 3], padding='same', name='CONV3', kernel_regularizer=regularizer)
    conv3_bn = tf.layers.batch_normalization(inputs=conv3, axis=-1, momentum=0.9, name="CONV3_BN",
                                             training=mode == tf.estimator.ModeKeys.TRAIN)
    conv3_pl = tf.layers.average_pooling2d(inputs=conv3_bn, padding='same', pool_size=[2, 2], name='CONV3_PL',
                                           strides=2)

    conv4 = tf.layers.conv2d(inputs=conv3_pl, filters=64, kernel_size=[3, 3], padding='same', name='CONV4', kernel_regularizer=regularizer)
    conv4_bn = tf.layers.batch_normalization(inputs=conv4, axis=-1, momentum=0.9, name="CONV4_BN",
                                             training=mode == tf.estimator.ModeKeys.TRAIN)
    conv4_pl = tf.layers.average_pooling2d(inputs=conv4_bn, padding='same', pool_size=[2, 2], name='CONV4_PL',
                                           strides=2)

    conv5 = tf.layers.conv2d(inputs=conv4_pl, filters=128, kernel_size=[1, 1], padding='same', name='CONV5', kernel_regularizer=regularizer)
    conv5_bn = tf.layers.batch_normalization(inputs=conv5, axis=-1, momentum=0.9, name="CONV5_BN",
                                             training=mode == tf.estimator.ModeKeys.TRAIN)
    conv5_pl = tf.layers.average_pooling2d(inputs=conv5_bn, padding='valid', pool_size=[21, 21], name='CONV5_PL',
                                           strides=1)


    dense = tf.reshape(conv5_pl, [-1, 128])
    Dense = tf.layers.dense(inputs=dense, units=128, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=Dense, units=2)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # nlabel = tf.cast(tf.one_hot(labels, 2), tf.int32)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels , logits=logits)

    nlbl = tf.reshape(tf.cast(tf.one_hot(labels, 2), tf.int32), [-1, 2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.nn.softmax(logits)).shape)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=nlbl,logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


model = tf.estimator.Estimator(model_fn=model_fn, model_dir='regularized_validation/')
# feature, label = importdata()
# sess = tf.Session()
# print(sess.run(label))
train_data = 'Newdataset/TFRecords/train.tfrecords'
valid_data = 'Newdataset/TFRecords/valid.tfrecords'

model.train(input_fn=lambda: importdata(train_data, 5))
eval = model.evaluate(input_fn=lambda : importdata(valid_data, 1))
print(eval)