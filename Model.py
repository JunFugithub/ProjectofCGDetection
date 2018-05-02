from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import numpy as np
import random
import os
import cv2

def HPF_5X5(x,W):
    return tf.nn.conv2d(x,W, strides=[1,2,2,1], padding='VALID')

def avg_pool_5x5(x):
    """Returns the result of average-pooling on input x with a 5X5 window"""
    return tf.nn.avg_pool(x, ksize=[1, 5, 5, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Model:

    def __init__(self, main_dir,batch_size, HPFtype):
        self.patch_size         =   650
        self.batch_size         =   batch_size
        self.seed               =   42
        self.kernal_5x5_1 = np.array([[0, 0, 0, 0, 0],
                                      [0, 0, -1, 0, 0],
                                      [0, -1, 4, -1, 0],
                                      [0, 0, -1, 0, 0],
                                      [0, 0, 0, 0, 0]])

        self.kernal_5x5_2 = np.array([[0, 0, 0, 0, 0],
                                      [0, -1, 2, -1, 0],
                                      [0, 2, -4, 2, 0],
                                      [0, -1, 2, -1, 0],
                                      [0, 0, 0, 0, 0]])

        self.kernal_5x5_3 = np.array([[-1, 2, -2, 2, -1],
                                      [2, -6, 8, -6, 2],
                                      [-2, 8, -12, 8, -2],
                                      [2, -6, 8, -6, 2],
                                      [-1, 2, -2, 2, -1]])

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("CG vs PG Model")
        # self.project_name       = input("pick a name for this project") + '/'


        print(" detect the file path and folder ...")
        self.main_dir           =   main_dir
        self.dataset_dir        =   main_dir+'Newdataset/'
        self.data_summary_dir   =   main_dir+'Datasummary/'
        self.tfrecords          =   self.dataset_dir+'TFRecords/'

        # 如果存在数据库文件直接读取，如果没有patch转为数据库文件
        self.file_name_set = {
            0: 'train.tfrecords',
            1: 'test.tfrecords',
            2: 'valid.tfrecords'
        }
        if not os.path.exists(self.tfrecords):
            print(" load patch then save it as data format ...")
            self.train_path     =   self.dataset_dir+'train/'
            self.test_path      =   self.dataset_dir+'test/'
            self.valid_path     =   self.dataset_dir+'validation/'
            self.loadPatchAndSave([self.train_path, self.test_path, self.valid_path])
        else:
            HPF = {
                'NOHPF': self.noHPF,
                'HPF1': self.HPF1,
                'HPF3': self.HPF3
            }
            print(" load dataset ...")
            self.hpf = HPF[HPFtype]
            self.createGraph()

        # if self.makeDir():
        #     print(" load patch then save it as data format ...")
    def createGraph(self):
        # for e in range(3):
        #     for b in range(int(nb / FLAGS.batch_size)):
        #         # print(sess.run(feature).shape)
        #         start = time.time()
        #         se
        #         print("step {}: with {} sec.".format(b + 1, time.time() - start))

        # with tf.Session() as sess:
        #     feature, label = self.loadData()
        #     x = self.hpf(feature)
        sess = tf.Session()
        start = time.time()
        feature, label = sess.run(self.loadData())
        print(feature.shape)
        print(" with {} sec.".format( time.time() - start))



            # feature, label = self.loadData()
            # x = self.hpf(tf.reshape(feature,[-1, 650,650, 1]))
            # print(sess.run(x).shape)
        pass

    def noHPF(self,X):
        return avg_pool_5x5(X)

    def HPF1(self, X):
        self.kernal_5x5_3 = self.kernal_5x5_3.reshape([5, 5, 1, 1])
        # self.trainingimages = HPF_5X5(trainingimages, self.kernal_5x5_3)
        return HPF_5X5(X, self.kernal_5x5_3)

    def HPF3(self):
        pass


    def parser(self, record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) # [self.patch_size, self.patch_size])
        label = tf.cast(parsed['label'], tf.int32)
        return image,label

    def loadData(self):
        # for file in filenames:

        file = [ self.tfrecords+ self.file_name_set[2] ]

        dataset = tf.data.TFRecordDataset(file).map(self.parser).shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        iterator= dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()

        return tf.reshape(feature,[-1, self.patch_size, self.patch_size, 1]) ,label #
        # feature, label = sess.run(iterator.get_next())
        # return feature.reshape(-1, self.patch_size, self.patch_size, 1), label # label shape = [batch_size, 1]

        # print(feature.reshape(-1,650,650,1).shape)
        # print(label.shape)

        # print(feature.shape)
        # desktop = '/Users/mac/Desktop/testnew.bmp'
        # cv2.imwrite(desktop,feature[10].reshape(650, 650, 1))
        #
        # print(label.shape)
        # print(label[10])



    def extractGreenChannel(self,image):
        return image[:,:,1]


    def loadPatchAndSave(self, paths):
        '''
        :param paths: train,test,valid load_path
        ::
        '''
        os.mkdir(self.tfrecords)
        # intersess = tf.InteractiveSession()

        for idx,path in enumerate(paths):   # 3次循环，train，valid，test
            # 读取图片并以二进制字符串的形式存入tfrecord中
            with tf.python_io.TFRecordWriter(self.tfrecords + self.file_name_set[idx]) as writer: # /TFRecords/train(valid, test).tfrecords
                for indx,name  in enumerate(['CGG/', 'Real/']):

                    start = time.time()
                    folder_path = path+name # train/CGG(Real)/
                    img_list    = [ folder_path + x for x in os.listdir(folder_path)]

                    number = 0
                    for f in img_list:
                        image = self.extractGreenChannel(cv2.imread(f))
                        # image = cv2.imread(f)
                        img_raw = image.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': _int64_feature(int(indx)),
                            'image_raw': _bytes_feature(img_raw)
                        }))
                        print(" name:{} -> label:{}".format(f,indx))
                        writer.write(example.SerializeToString())
                        number+=1
                    print(" nb:{}".format(number))
                    print(" {}      loaded with {:.4f} sec. ".format(folder_path,time.time() - start))


    # def makeDir(self):
# self.tfdata_path = self.data_summary_dir + 'TFDataset/'
#     self.project_tfdata = self.tfdata_path + self.project_name
#     self.tfsumm_path = self.data_summary_dir + 'TFSummary/'
#     self.project_tfsumm = self.tfsumm_path + self.project_name
#     if os.path.exists(self.project_tfdata):
#         self.loadData()
#         return False
#
#
#     if not os.path.exists(self.data_summary_dir):
#         print("     make necessary folder ...")
#         os.mkdir(self.data_summary_dir)
#
#         os.mkdir(self.tfdata_path)
#         os.mkdir(self.project_tfdata)
#
#         os.mkdir(self.tfsumm_path)
#         os.mkdir(self.project_tfsumm)
#
#         return True


if __name__ == '__main__':
    HPF = ['NOHPF', 'HPF1', 'HPF3']
    main_dir = '/Users/mac/Documents/Project_of_Graduation/formal_tech/'
    batch_size = 32
    model    = Model(main_dir,batch_size, HPF[1])