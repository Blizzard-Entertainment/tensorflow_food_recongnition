from app_demo.request_utils.common import *
from app_demo.models import *
from app_demo.utils.img_handle import *
import base64
import os
import re
from app_demo.predict.predict_img import *


def query_img_predict(request):
    data = request.POST
    if check_param_error(data, 'img_base64'):
        return check_param_error(data, 'img_base64')
    img_base64 = data['img_base64']
    img_name = data['img_name']
    response = {}
    response['status'] = 200
    response['img_name'] = img_name
    response['predict_result'] = predict_img(img_base64)
    return response

# def save_img(request):
#     urls = ''
#     dir_name = date.today().__str__().replace('-', '_', 2) # 2019_06_21
#     dirs = os.path.join(IMAGE_ROOT, dir_name) # 将日期作为目录名
#     if not os.path.isdir(dirs):
#         os.makedirs(dirs) # 判断目录是否存在，不存在则创建
#     for img in request.data['imgs']:
#         strs = img.split(',')
#         suffix = re.findall(r'/(\w+?);', strs[0])[0] # 取得文件后缀
#         # 拼接服务器上的文件名
#         # datetime.now()取得当前时间，精确到了微秒，一般来说是唯一的了，因为目录是日期，所以文件名就去掉日期，最后会是一串数字
#         img_name = re.sub(r':|\.', '', datetime.now().__str__().split(' ')[1]) + '.' + suffix 
#         img_path = os.path.join(dirs, img_name)
#     with open(img_path, 'wb') as out:
#         out.write(base64.b64decode(strs[1])) # base64解码，再写入文件
#         out.flush()
#         urls += os.path.join(WEB_HOST_MEDIA_URL, dir_name, img_name) + '[/--sp--/]' # 拼接URL，URL与URL之间用[/--sp--/]隔开
#     result = {}
#     result['status'] = status.HTTP_200_OK
#     result['message'] = '图片上传成功'
#     result['urls'] = urls[:len(urls) - len('[/--sp--/]')] # 去掉末尾的[/--sp--/]
#     return Response(data=result)

# def save_img_2():
#     qrcode = sample()
#     f = BytesIO()
#     f.write(qrcode)
#     # InMemoryUploadedFile 需要的参数：file, field_name, name, content_type, size, charset, content_type_extra = None
#     image = InMemoryUploadedFile(f, None, "{}_{}.png".format('KF45464789',random.randint(100,999)), None, len(qrcode), None, None)
#     instance.qr_code_url = image
#     instance.save()