from django.urls import path
from .consumers import ChatConsumer


channel_routing = [
    path('', ChatConsumer)]
