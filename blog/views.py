from blog.VocabGen import vocabexecute
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin,UserPassesTestMixin
from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)


from django.http import HttpResponse
from .models import Post
# Create your views here.

def home(request):
    context = {
        'posts':Post.objects.all()
    }
    return render(request,'blog/home.html',context)

class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html' #<app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']

class PostDetailView(DetailView):
    model = Post

class PostCreateView(LoginRequiredMixin,CreateView):
    model = Post
    fields = ['title','content']
    
    def form_valid(self,form):
        form.instance.author = self.request.user
        return super().form_valid(form)

class PostUpdateView(LoginRequiredMixin,UserPassesTestMixin,UpdateView):
    model = Post
    fields = ['title','content']
    
    def form_valid(self,form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

class PostDeleteView(LoginRequiredMixin,UserPassesTestMixin,DeleteView):
    model = Post
    success_url = '/'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False


def about(request):
    return render(request,'blog/about.html',{'title':'About'})

from .autosearch import autosearch
from .OurMcqGen import excecute
def mcqview(request):
    context = dict()
    if request.method == "POST":
        if "mcq_content_form" in request.POST:
            compression_ratio = int(request.POST.get("compression_ratio"))/100
            print("compression_ratio: ",compression_ratio)
            str = request.POST.get("full_text")
            mcqs = excecute(str,compression_ratio)
            context = {'mcqs':mcqs}
            context['full_text'] = str
            return render(request,'blog/Mcq_Steps.html',context)
        elif "mcq_search_bar_form" in request.POST:
            str = request.POST.get("search_bar")
            result = autosearch(str)
            searched = {'result':result}
            return render(request,'blog/Mcq_Steps.html',searched)
    else:
        return render(request,'blog/Mcq_Steps.html')

def summaryview(request):
    if request.method == "POST":
        if "mcq_content_form" in request.POST:
            compression_ratio = int(request.POST.get("compression_ratio"))/100
            print("compression_ratio: ",compression_ratio)
            str = request.POST.get("full_text")
            summary = excecute(str,compression_ratio,True)
            context = {'summary':summary}
            return render(request,'blog/Summary.html',context)
        elif "mcq_search_bar_form" in request.POST:
            str = request.POST.get("search_bar")
            result = autosearch(str)
            searched = {'result':result}
            return render(request,'blog/Summary.html',searched)
    else:
        return  render(request,'blog/Summary.html')

from .VocabGen import vocabexecute
def vocabview(request):
    if request.method == "POST":
        if "vocab_content_form" in request.POST:
            type_of_grammar = request.POST.getlist('type_of_grammar[]')
            print(type_of_grammar)
            str = request.POST.get("full_text")
            vocabmcqs = vocabexecute(str,type_of_grammar)
            context = {'vocabmcqs':vocabmcqs}
            return render(request,'blog/vocab.html',context)
        elif 'vocab_search_bar_form' in request.POST:
            str = request.POST.get("search_bar")
            result = autosearch(str)
            searched = {'result':result}
            return render(request,'blog/vocab.html',searched)
    else:
        return  render(request,'blog/vocab.html')

def indexview(request):
    return render(request,'blog/index.html')









