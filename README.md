#Pythonic OCR
[Website](https://pythonic-ocr.herokuapp.com/)

A convolutional neural network implemented in pure numpy.

Description
-----------
I built this project to teach myself about how deep convolutional neural networks function. First, I created my own MNIST-like dataset with about 30 alphanumeric symbols.

Next, I trained a deep convolutional network using Keras and saved the weights using python's pickle utility. Finally, I rewrote all the forward propagation code in pure numpy (as opposed to Theano or Tensorflow as in Keras). This lets me run the network as a demo via heroku.

There are a few Easter eggs too: try a five-pointed star or a triangle!

Dependencies
--------
*All code is written in python 3.

Dependencies are packaged in the flask folder, so this app does not have any external depencies.

Examples
--------
I trained the network on my own handwriting, so performace will be slightly different for other individuals. Even so, it is good at making fine discrimination such as judging the difference between O/Q:

![recognized_o.png](https://github.com/greydanus/pythonic_ocr/blob/master/app/static/img/recognized_o.png) ![recognized_q.png](https://github.com/greydanus/pythonic_ocr/blob/master/app/static/img/recognized_q.png)
