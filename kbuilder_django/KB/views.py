from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.core.files.storage import FileSystemStorage

from .forms import DocumentForm
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import sys
from io import StringIO
from .websocket_send import send_text
from kbuilder.src.KB_funcs import KB_Bot
from kbuilder.src.utils import Config
import asyncio

config = Config()
bot = KB_Bot()
event_loop = asyncio.new_event_loop()
#channel_layer = get_channel_layer()
#async_to_sync(channel_layer.group_send)('message_group', {'type':'receive',\
#             'message': "done"})
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def logs(request):
    """Returns `JsonResponse` object"""

    # you can change the request method in the following condition.
    # I dont know what you're dealing with.
    if request.is_ajax() and request.method == 'GET':
        # main logic here setting the value of resp_data

        resp_data = {
            'html': 'stuff',
            # more data
        }

        return JsonResponse(resp_data, status=200)

def index(request):
    if request.method == 'POST':
        print("POST", request.POST)
        if 'file-upload' in request.POST:
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                obj = form.save()
                send_text("Processing, this may take a while, please wait...", event_loop)
                with open(obj.document.path, 'r', encoding='utf8') as f:
                    preview = f.read()[:300]
                config.text_file = obj.document.path
                bot.__init__(args=config, restart=True)
                print("filename: ", bot.filename)
                print("modelname: ", config.state_dict)
                form = DocumentForm()
                return render(request, 'KB/index.html', {'preview': preview,\
                                                         'form': form,\
                                                         'filename': bot.filename,\
                                                         'modelname': config.state_dict,\
                                                         'message': "%s uploaded and processed. Please load model with button below." % bot.filename})
        
        elif 'query' in request.POST:
            query = request.POST.get('query-text', None)
            ans = bot.ask(query)
            send_text("Done", event_loop)
            return render(request, 'KB/index.html', {'ans': ans,\
                                                     'filename': bot.filename,\
                                                     'message': "Query processed."})
    
        else:
            form = DocumentForm()
            return render(request, 'KB/index.html', {'form': form,\
                                                     'message':'Error please try again'})
    else:
        form = DocumentForm()
        return render(request, 'KB/index.html', {'form': form,\
                                                 'message': "Please upload file below"})
            
 
def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'KB/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'KB/simple_upload.html')

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('KB')
    else:
        form = DocumentForm()
    return render(request, 'KB/model_form_upload.html', {
        'form': form
    })
    

