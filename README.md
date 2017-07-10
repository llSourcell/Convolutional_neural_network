# Convolutional Neural Network

## Overview

This is the code for [this](https://youtu.be/FTr3n7uBIuE) video on Youtube by Siraj Raval as part of The Math of Intelligence course. A convolutional neural network implemented in pure numpy. It uses a MNIST-like dataset with about 30 alphanumeric symbols. The author trained a deep convolutional network using Keras and saved the weights using python's pickle utility. Only the the forward propagation code is rewritten in pure numpy (as opposed to Theano or Tensorflow as in Keras). Which lets us run the network as a demo via heroku. For backpropagation in numpy for a convnet see [this](https://github.com/Kankroc/NaiveCNN)

![recognized_o.png](https://github.com/greydanus/pythonic_ocr/blob/master/app/static/img/recognized_o.png) ![recognized_q.png](https://github.com/greydanus/pythonic_ocr/blob/master/app/static/img/recognized_q.png)

Live web app is here:
[Website](https://pythonic-ocr.herokuapp.com/)


## Dependencies
--------

Dependencies are packaged in the flask folder, so this app does not have any external depencies. Run `pip install -r requirements.txt` to install them. 

Install pip [here](https://pip.pypa.io/en/stable/). 


## Usage

to start the web app run `python run.py` . To start the notebook run `jupyter notebook` in terminal. 

Install jupyter [here](http://jupyter.readthedocs.io/en/latest/install.html). 


## Credits 

Credits for this code go to [greydanus](https://github.com/greydanus/pythonic_ocr). I've merely created a wrapper to get people started. 

