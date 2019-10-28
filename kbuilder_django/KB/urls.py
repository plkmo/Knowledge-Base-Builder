from django.urls import path

from . import views

app_name = 'KB'
urlpatterns = [
    # ex: /KB/
    path('', views.index, name='index'),
    # ex: /KB/5/
    path('<int:question_id>/', views.detail, name='detail'),
    # ex: /KB/5/results/
    path('<int:question_id>/results/', views.results, name='results'),
    # ex: /KB/5/vote/
    path('<int:question_id>/vote/', views.vote, name='vote'),
]
