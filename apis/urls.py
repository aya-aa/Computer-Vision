
from django.urls import path 
from .views import imageapi, videoapi

urlpatterns = [
    path("image-upload/", imageapi),
    path("video-upload/", videoapi)
]


    
