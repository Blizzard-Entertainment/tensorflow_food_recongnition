from divide_dataset import *
from create_tf_record import *
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

print("[INFO]@File   : devide dataset DONE")

# 参数设置

resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
shuffle=True
log=5
dataset_file = dataset
# 产生train.record文件
train_labels = '{}/train.txt'.format(dataset_file)  # 图片路径
mkdir("{}/record".format(dataset_file))
train_record_output = '{}/record/train{}.tfrecords'.format(dataset_file, resize_height)
create_records(train_labels, train_record_output, resize_height, resize_width,shuffle,log)
train_nums=get_example_nums(train_record_output)
print("save train example nums={}".format(train_nums))

# 产生val.record文件
val_labels = '{}/val.txt'.format(dataset_file)  # 图片路径
val_record_output = '{}/record/val{}.tfrecords'.format(dataset_file, resize_height)
create_records(val_labels, val_record_output, resize_height, resize_width,shuffle,log)
val_nums=get_example_nums(val_record_output)
print("save val example nums={}".format(val_nums))

# 测试显示函数
# disp_records(train_record_output,resize_height, resize_width)
# batch_test(train_record_output,resize_height, resize_width)
