
import matplotlib.pyplot as plt
import numpy as np
import json
import time

# date_time = str(time.asctime(time.localtime(time.time())).replace(':','-'))
date_time = 'Mon May  4 00-17-12 2020'
datafile = 'logs/loss_acc_{}.txt'.format(date_time)
plotFile = 'logs/plot/plot_{}.png'.format(date_time)
plotaccFile = 'logs/plot/plot_acc_{}.png'.format(date_time)
plotlossFile = 'logs/plot/plot_loss_{}.png'.format(date_time)

with open(datafile, 'a') as f:
    pass

def generate_data_dict(i_step=0, train_loss=0, train_acc=0, val_loss=0, val_acc=0):
    data = {}
    i_step = str(i_step)
    data[i_step] = {}
    data[i_step]['train_loss'] = train_loss
    data[i_step]['train_acc'] = train_acc
    data[i_step]['val_loss'] = val_loss
    data[i_step]['val_acc'] = val_acc
    print(data)
    return data
    

def show_data_from_file(filename = datafile):
    dic = {}
    with open(filename, 'r') as f:
        lines = f.read()
        if lines != '':
            dic = eval(lines)   #读取的str转换为字典

    plt.figure()
    lines_acc_label = ['train_acc', 'val_acc' ]
    lines_loss_label = ['train_loss','val_loss']
    lines_color = ['red', 'blue']
    plt.subplot(121)

    for label in lines_acc_label:
        label_tuple = []
        x = []
        y = []

        for (k,v) in dic.items():
            label_tuple.append((int(k),float(v[label])))

        label_tuple = sorted(label_tuple)
        for t in label_tuple:
            x.append(t[0])
            y.append(t[1])

        plt.plot(x, y, color =lines_color[lines_acc_label.index(label)], label = label)

    plt.ylim(0,1)
    plt.xlabel('iteration')
    plt.ylabel('acc/loss')
    plt.legend(loc ='best')
    plt.subplot(122)

    for label in lines_loss_label:
        label_tuple = []
        x = []
        y = []

        for (k,v) in dic.items():
            label_tuple.append((int(k),float(v[label])))

        label_tuple = sorted(label_tuple)
        for t in label_tuple:
            x.append(t[0])
            y.append(t[1])

        plt.plot(x, y, color =lines_color[lines_loss_label.index(label)], label = label)


    plt.ylim(0,100)
    plt.xlabel('iteration')
    plt.ylabel('acc/loss')
    plt.legend(loc ='best')
    plt.savefig(plotFile)
    plt.show()


def write_data_to_file(filename, data):
    dic = {}
    with open(filename, 'r') as f:
        lines = f.read()
        if lines != '':
            dic = eval(lines)

    with open(filename, 'w') as f:
        dic.update(data) 
        f.write(str(dic))

def feed_data_to_file(i_step=2, train_loss=0, train_acc=0, val_loss=0, val_acc=0):
    data = generate_data_dict(i_step, train_loss, train_acc, val_loss, val_acc)
    write_data_to_file(datafile, data)

if __name__ == '__main__':
    show_data_from_file()


