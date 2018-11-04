# Digit Recognizer
Recognizes and predicts Hindu Arabic digit.

## Short Description:
Here, users can draw a digit in the web page, which is implemented by HTML canvas, and when predict button is pressed, the neural network predicts the digit.  The neural network is written from scratch and the parameters(weights and bias) are save in pickle files.  The parameters are loaded and prediction is done on the input image in the real time.

## Requirements:
- Python 3
- Numpy
- Django 1.11

## Dependencies
 Install Django and Numpy by executing the following commands in the terminal.
```
$ sudo apt-get install python-pip  
$ sudo pip install numpy scipy
$ sudo pip install django==1.11

```

## Instructions:
* Run ./manage.py with python3 as the interpreter(shebang would take care in our file) and the server would start running.
```
./main.py
```
* Click the [link](http://127.0.0.1:8000)(http://127.0.0.1:8000) and the web application would start running.

## Further Enhancements:
* We have used deep neural network architecture here.  We can use Convolutional Neural Network for AI which would increase accuracy of our prediction.
* We can augment our training data to account for slightly rotated and dilated images.