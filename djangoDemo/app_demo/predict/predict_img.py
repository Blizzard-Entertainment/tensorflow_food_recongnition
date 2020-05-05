#coding=utf-8
# 使用说明
# 修改modelpath指定为训练的模型
# 修改模型预测
# 修改图像路径
# 修改类别种类
import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import app_demo.predict.slim.nets.inception_v3 as inception_v3
import tensorflow.contrib.slim as slim
from app_demo.predict.create_tf_record import *
from app_demo.utils.img_handle import *

def load_graph(models_path,labels_filename,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t', encoding='utf-8')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    #其他模型预测请修改这里
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    sess_param = [score, class_id]
    return sess, sess_param, input_images,labels

def init_graph():
    class_nums=50
    labels_filename="{base_path}/label_map.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    models_path="{base_path}/models/model.ckpt-45000".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    return load_graph(models_path, labels_filename, class_nums, data_format)

SESS, SESS_PARAM, INPUT_IMAGES,LABELS = init_graph()

def predict_img(img_base64, method ='base64'):
    resize_height = 224
    resize_width = 224
    # img_base64 = sample()
    img = read_image_base64(img_base64, resize_height, resize_width, normalization=True)
    img=img[np.newaxis,:]
    pre_score,pre_label = SESS.run(SESS_PARAM, feed_dict={INPUT_IMAGES:img})
    max_score=pre_score[0,pre_label]
    predict_info = "{} is: pre labels:{},name:{} score: {}".format('img',pre_label,LABELS[pre_label], max_score)
    return predict_info

def predict_single_image(models_path,image_dir,labels_filename,labels_nums, data_format):
    tf.reset_default_graph()
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t', encoding='utf-8')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    #其他模型预测请修改这里
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)

    im=read_image(image_dir,resize_height,resize_width,normalization=True)
    im=im[np.newaxis,:]
    pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
    max_score=pre_score[0,pre_label]
    predict_info = "{} is: pre labels:{},name:{} score: {}".format(image_dir,pre_label,labels[pre_label], max_score)
    # print(predict_info)
    sess.close()
    return predict_info


def custom_predict(params={}):
    class_nums=50
    image_dir='test_image'
    labels_filename="{base_path}/label_map.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    image_path = params['image_path'] or "{base_path}/test_image/pic_16.jpg".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    # map_labels_filename='dataset/label.txt'
    models_path="{base_path}/models/model.ckpt-45000".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    # predict_list_image(models_path,image_dir, labels_filename, class_nums, data_format)
    predict_info = predict_single_image(models_path,image_path, labels_filename, class_nums, data_format)
    # predict_info = 'good'
    return predict_info
    
if __name__ == '__main__':

    custom_predict()
