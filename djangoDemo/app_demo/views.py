from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from app_demo.request_utils.request_response import *
from app_demo.models import *
import json
"""
 django.http模块中定义了HttpResponse 对象的API
 作用：不需要调用模板直接返回数据
 HttpResponse属性：
    content: 返回内容,字符串类型
    charset: 响应的编码字符集
    status_code: HTTP响应的状态码
"""

"""
hello 为一个视图函数，每个视图函数必须第一个参数为request。哪怕用不到request。
request是django.http.HttpRequest的一个实例
"""
def request_test(request):
    if request.method == 'POST':
        return JsonResponse(query_img_predict(request))
    return None


def request_img_predict(request):
    if request.method == 'POST':
        # name = "fake_name.jpg"
        # fake_string = sample()
        # img = str2imgbase64(fake_string)
        # if img != None:
        #     name = request.FILES.get('img').name
        #     new_img = IMG(
        #         img = img,
        #         name = name
        #     )
        #     new_img.save()
        #     predict_result = custom_predict({'image_path' : "media/img/{}".format(new_img.name)})
        #     response = {"result":"predict_result"}
        return HttpResponse()




def predict_pic(request):
    predict_result = custom_predict()
    return HttpResponse(predict_result)

def uploadImg(request):
    # print('[INFO]uploadImg')
    if request.method == 'POST':
        # print("[INFO]request:{}".format(request))
        img = request.FILES.get('img')
        if img != None:
            name = request.FILES.get('img').name
            new_img = IMG(
                img = img,
                name = name
            )
            new_img.save()
            predict_result = custom_predict({'image_path' : "media/img/{}".format(new_img.name)})
            return HttpResponse(predict_result)
        # img =request.FILES.get('img', None)
    return render(request, 'uploading.html')


def showImg(request):
    imgs = IMG.objects.all()
    content = {
        'imgs':imgs,
    }
    for i in imgs:
        print (i.img.url)
    return render(request, 'showing.html', content)

def hello(request):
    return HttpResponse('Hello World')

def request_hello(request):
    if request.method == 'POST':
        # print("request.body:{}".format(request.GET))
        return JsonResponse({'data':'hello'})
    return HttpResponse(404)

