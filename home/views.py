from django.http import HttpResponse
from django.shortcuts import render
from RetinopathyModel.model import predict

# Create your views here.
def home(request):
    return render(request, "home.html")


def upload(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        pred = predict(upload)
        data = {"rating" : pred[0], "state" : pred[1]}
        return render(request, "predict.html", context=data)
    return render(request, "upload.html")