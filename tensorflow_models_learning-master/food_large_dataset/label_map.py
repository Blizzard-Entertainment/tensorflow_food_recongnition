import os
import os.path

# 由于读入label的时候没有按顺序读入 导致使用文件名读入的时候出现了乱序的问题 使用这个map能够生成对应顺序的txt文件以确保标签的准确性

id_path = "label.txt"
origin_path = "label_origin.txt"
map_path = "label_map.txt"

def read_txt(file_name):
    list_txt = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            list_txt.append(line)
    
    return list_txt

def read_origin():
    return read_txt(origin_path)

def read_id_origin():
    list_txt = read_txt(id_path)
    for i in range(0, len(list_txt)):
        list_txt[i] = int(list_txt[i]) - 1
    return list_txt

def read_map_origin():
    return read_txt(map_path)




def generate_map_list(origin_list, id_list):
    map_list = []
    for i in range(0, len(id_list)):
        map_list.append(origin_list[id_list[i]])
    return map_list

def write_list_to_txt(a_list, file_name):   
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in a_list:
            f.write(i+'\n')



origin_list = read_origin()
id_list = read_id_origin()
if __name__ == "__main__":
    
    map_list = generate_map_list(origin_list, id_list)
    write_list_to_txt(map_list, map_path)
