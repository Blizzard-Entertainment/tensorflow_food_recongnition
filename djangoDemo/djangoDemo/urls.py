"""djangoDemo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from app_demo import views
from django.views.generic import RedirectView
urlpatterns = [
    # path('admin/', admin.site.urls),
    url(r'^hello/$', views.hello),
    url(r'^request_test/$', views.request_test),
    url(r'^request_hello/$', views.request_hello),
    url(r'^request_img_predict/$', views.request_img_predict),
    # url(r'^predict/$', views.predict_pic),
    url(r'^upload', views.uploadImg),
    url(r'^show', views.showImg),
    url(r'^$', RedirectView.as_view(url='hello/')),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
