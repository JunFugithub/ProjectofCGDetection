import os
import tensorflow as tf
import cv2
from PIL import Image

pwd      = '/Users/mac/Documents/Project_of_Graduation/formal_tech/Newdataset/test/CGG/'
# writer   = tf.python_io.TFRecordWriter("/Users/mac/Documents/Project_of_Graduation/formal_tech/Datasummary/TFSummary/train_tfrecords")
example1 = '/Users/mac/Documents/Project_of_Graduation/formal_tech/Newdataset/test/CGG/level_design_6681#0000.bmp'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# img_raw = cv2.imread(example1).tostring()
# print(img_raw)

# with tf.python_io.TFRecordWriter('train_test.tfrecords') as writer:
#     for file in os.listdir(pwd):
#         file    = pwd + file
#         img_raw = cv2.imread(file).tostring()
#
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'image_raw': _bytes_feature(img_raw)
#         }))
#         writer.write(example.SerializeToString())
def parser(record):
    keys_to_features={
        'image_raw': tf.FixedLenFeature((), tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    return image

filename = ['train_test.tfrecords']
sess     = tf.InteractiveSession()
dataset  = tf.data.TFRecordDataset(filename)
dataset  = dataset.map(parser)
dataset  = dataset.batch(3)
# dataset  = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
#
img      = iterator.get_next()
img = sess.run(img)
# img = img.reshape(650,650,3)
print(img.shape)
#
#
# desktop = '/Users/mac/Desktop/test1.bmp'
# cv2.imwrite(desktop,img)



# sess = tf.Session()
# inc_dataset = tf.data.Dataset.range(30)
# dec_dataset = tf.data.Dataset.range(0, -100, -1)
# dataset = inc_dataset
# batched_dataset = dataset.batch(14)
#
# iterator = batched_dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
# print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
# print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
# batched_dataset.repeat()


'''
Save as .npy
'''
# for path in paths:
#     for cls in ['CGG/', 'Real/']:
#         start = time.time()
#
#         load_path = path + cls
#         fileroad = [load_path + x for x in os.listdir(load_path)]
#         image = np.empty([len(fileroad), self.patch_size, self.patch_size])
#         for i in range(len(fileroad)):
#             image[i] = self.extractGreenChannel(cv2.imread(fileroad[i]))
#         image = image.reshape([len(fileroad), self.patch_size, self.patch_size, 1])
#         np.save(self.project_tfdata + path.split('/')[-2] + '_' + cls.split('/')[0] + '.npy', image)
#         print(" {}      loaded with {:.4f} sec. And the shape of \"{}\" is {}".format(load_path, time.time() - start,
#                                                                                       path.split('/')[-2] + '_' +
#                                                                                       load_path.split('/')[-2],
#                                                                                       image.shape))