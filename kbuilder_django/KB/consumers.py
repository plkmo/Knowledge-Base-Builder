#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:16:55 2019

@author: weetee
"""
import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from kbuilder.src.KB_funcs import KB_Bot
from kbuilder.src.utils import Config

config = Config()
bot = KB_Bot()

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.user = self.scope["user"]; print(self.user)
        self.accept()
        async_to_sync(self.channel_layer.group_add)("message_group", self.channel_name)

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)("message_group", self.channel_name)
        pass

    def receive(self, text_data):
        handler_dict = {"filename":"load_model", "query":"query", "triplet":"find_triplet"}
        print("TEXT_DATA:_received", text_data)
        text_data_json = json.loads(text_data)
        message = text_data_json['text']
        label = text_data_json['label']
        
        # Send message to room group
        async_to_sync(self.channel_layer.group_send)(
            "message_group",
            {
                'type': handler_dict.setdefault(label,'send_message'),\
                'text': message,\
                'label': label
            }
        )
        
    def send_message(self, event):
        print("TEXT_DATA_send:", event)

        # Send message to WebSocket
        
        self.send(text_data=json.dumps(event))
        
        #self.send(message)
        
    def load_model(self, event):
        print("TEXT_DATA_load_model:", event)
        config.state_dict = event['text']
        bot.load_(config)
        self.send(text_data=json.dumps({"type":"msg",\
                                        "text": "Loaded model!",\
                                        "label": "model_status"}))
        
    def query(self, event):
        ans = bot.ask(event['text'])
        self.send(text_data=json.dumps({"type":"query",\
                                        "text": ans,\
                                        "label": "query"}))
        
    def find_triplet(self, event):
        triplet = event['text'].split("@")
        assert len(triplet) == 3
        sub, pred, obj = triplet
        matched_triplets = bot.query(triplet, ['subject', 'predicate', 'object'])
        self.send(text_data=json.dumps({"type": "triplet",\
                                        "text": str(matched_triplets),\
                                        "label": "triplet"}))