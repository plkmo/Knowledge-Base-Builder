from django.urls import path

from . import views

app_name = 'KB'
urlpatterns = [
    # ex: /KB/
    path('', views.index, name='index'),
    path('upload_simple/', views.simple_upload, name='simple_upload'),\
    path('upload_model/', views.model_form_upload, name='model_form_upload')
]
