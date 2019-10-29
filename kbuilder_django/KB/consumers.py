#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:16:55 2019

@author: weetee
"""
import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        async_to_sync(self.channel_layer.group_add)("message_group", self.channel_name)

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)("message_group", self.channel_name)
        pass

    def receive(self, text_data):
        print("TEXT_DATA:", text_data)
        self.send(json.dumps({
            "type": "websocket.send",
            "text": text_data["message"],
        }))