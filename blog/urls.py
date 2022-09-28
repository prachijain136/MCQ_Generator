from django.urls import path
from . import views
from .views import (PostListView,PostDetailView,PostCreateView,
                        PostUpdateView,PostDeleteView, indexview, mcqview)

urlpatterns = [
    path('', PostListView.as_view(),name='blog-home'),
    path('post/<int:pk>', PostDetailView.as_view(),name='post-detail'),
    path('post/new', PostCreateView.as_view(),name='post-create'),
    path('about/',views.about,name='blog-about'),
    path('post/<int:pk>/update/', PostUpdateView.as_view(),name='post-update'),
    path('post/<int:pk>/delete/', PostDeleteView.as_view(),name='post-delete'),
    path('mcqsteps/', views.mcqview,name='mcq-gen'),
    path('index/', views.indexview,name='mcq-index'),
    path('summary/',views.summaryview,name='summary-gen'),
    path('vocab/',views.vocabview,name='vocab-gen'),

]

