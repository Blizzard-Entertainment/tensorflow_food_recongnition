# -*-coding:utf-8-*-


import os
import os.path
import random
DIVIDE_TRAIN_PERCENT = 0.7

def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)

def write_list(content, filename, mode = 'w'):
    with open(filename, mode) as f:
        for line in content:
            f.write(line + '\n')

def get_files_label(dir):
    files_list = os.listdir(dir)
    return files_list

def get_files_list(dir, label_list):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    train_list = []
    val_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            # print("dirnames is: " )
            # print(dir)
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]
            path_name = "{}/{}/{}".format(dir,curr_file, filename)
            # print(path)
            # parent_file = parent.split(os.sep)[-2]
            # print("curr_file :{}".format(parent_file))
            if curr_file in label_list:
                labels = label_list.index(curr_file)
            if random.random() < DIVIDE_TRAIN_PERCENT:
                train_list.append([path_name, labels])
            else:
                val_list.append([path_name, labels])
    return train_list, val_list

if __name__ == '__main__':


    dataset = "food_large_dataset_aug/"

    dataset_dir = dataset + 'dataset'
    train_txt = dataset + 'train.txt' 

    val_dir = dataset + 'dataset'
    val_txt = dataset + 'val.txt'

    label_txt = dataset + 'label.txt'
    label_list = get_files_label(dataset_dir)
    write_list (label_list, label_txt , mode='w')

    train_data, val_data = get_files_list(dataset_dir, label_list)

    write_txt(train_data, train_txt, mode='w')
    write_txt(val_data, val_txt, mode='w')

    print("[INFO]@File   : create_labels_files.py DONE")

