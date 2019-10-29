from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.core.files.storage import FileSystemStorage

from .forms import DocumentForm

from kbuilder.src.KB_funcs import KB_Bot
from kbuilder.src.utils import Config

config = Config()
bot = KB_Bot(config)

def index(request):
    if request.method == 'POST':
        print("POST", request.POST)
        if 'file-upload' in request.POST:
            form = DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                obj = form.save()
                #print("ID", form.instance.id)
                #print(obj.document.path)
                with open(obj.document.path, 'r', encoding='utf8') as f:
                    preview = f.read()[:100]
                config.text_file = obj.document.path
                bot.__init__(config)
                return render(request, 'KB/index.html', {'preview':preview})
        
        elif 'query' in request.POST:
            query = request.POST.get('query-text', None)
            ans = bot.ask(query)
            return render(request, 'KB/index.html', {'ans':ans})
    else:
        form = DocumentForm()
        return render(request, 'KB/index.html', {
            'form': form
            })
 
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
    

