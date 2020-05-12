from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import glob, os
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(file_name, input_shape, random=True, jitter=.7, hue=.1, sat=1.2, val=1.2, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    image = Image.open(file_name)
    iw, ih = image.size
    h, w = input_shape

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.7,2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上白色区域
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (255,255,255))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flip = rand()<.2
    if flip: image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    
    return image_data

def normal_(annotation_line, input_shape):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    return image

if __name__ == "__main__":
    size_params = [224, 224]
    
    dataset = "dataset"
    aug_max_times = 1

    for infile in glob.glob("./{}/*/*.jpg".format(dataset)):
        print("[INFO]Execute {}".format(infile))
        img_name = infile.split(os.sep)[-1].split('.')[0]
        filename = infile.split(os.sep)[-2]
        for i in range(0,aug_max_times):
            image_data = get_random_data(infile,size_params)
            img = Image.fromarray((image_data*255).astype(np.uint8))
            img.save("./{}/{}/{}_aug_{}.jpg".format(dataset, filename, img_name, i), quality=95)

    print("[INFO] imgsave")