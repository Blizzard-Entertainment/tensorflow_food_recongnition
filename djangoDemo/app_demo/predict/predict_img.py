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

def load_graph(models_path,labels_filename,label_id_filename,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t', encoding='utf-8')
    labels_id = np.loadtxt(label_id_filename, str, delimiter='\t', encoding='utf-8')
    labels_nums = labels_id.size
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
    return sess, sess_param, input_images, labels, labels_id

def init_graph():
    class_nums=50
    labels_filename="{base_path}/label_map.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    label_id_filename="{base_path}/label.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    models_path="{base_path}/models/model.ckpt-60000".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    batch_size = 1  #
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    return load_graph(models_path, labels_filename,label_id_filename, class_nums, data_format)

SESS, SESS_PARAM, INPUT_IMAGES, LABELS, LABELS_ID = init_graph()

def get_top_result(pre_score, labels, label_id, max_top = 3, score_threshod = 0.01):
    '''
    用于获取结果中预测分数较高的数个结果，返回result是一个dict
    pre_score：预测的分数
    labels：预测的标签列表
    max_top：最多返回几个
    score_threshod：返回的阈值 大于这个分数就会被返回 小于分数则舍弃
    
    '''
    pre_sort = np.argsort(-pre_score)[0]
    result = {}
    # 因为需要用json网络传输， 因此所有的数据类型都要转换成str json不认识float32
    for i in range(0,max_top):
        top_dict = {}
        top_dict['score'] = str(pre_score[0][pre_sort[i]])
        top_dict['label_name'] = str(labels[pre_sort[i]])
        top_dict['label_id'] = str(label_id[pre_sort[i]])
        if i == 0 or pre_score[0][pre_sort[i]] > score_threshod:
            result [i+1] = top_dict
    return result


def predict_img(img_base64, method ='base64'):
    resize_height = 224
    resize_width = 224
    img = read_image_base64(img_base64, resize_height, resize_width, normalization=True)
    img=img[np.newaxis,:]
    pre_score,pre_label = SESS.run(SESS_PARAM, feed_dict={INPUT_IMAGES:img})
    result = get_top_result(pre_score, LABELS, LABELS_ID)
    return result


