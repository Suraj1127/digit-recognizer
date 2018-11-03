from django.http import HttpResponse
from django.shortcuts import render
from .digit_predictor import DigitPredictor

def home(request):
    if request.method == 'POST':
        digit_predictor = DigitPredictor(request.POST['data_url'])
        predicted_digit = digit_predictor.predict_digit()
        return HttpResponse(predicted_digit)
    else:
        return render(request, 'canvas/home.html')