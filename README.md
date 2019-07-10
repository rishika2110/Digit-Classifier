# The MNIST Digit Classifier

## Introduction

The “hello world” of object recognition for machine learning and deep learning is the MNIST dataset for handwritten digit recognition.
The dataset consists of images of digits which were taken from a variety of sources.Each image is 28 by 28 pixel square(784 pixels total).
A standard split of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.
It is a digit recognition task.As such there are 10 digits(0 to 9) or 10 classes to predict.

## Description

We did the classification task in the following ways:
 1) Single layer Neural network (using only numpy)
           
 2) Multilayer Neural network (using only numpy)
           
 3) Convolutional Neural network (using pytorch)
 
 The results were displayed using the cost vs number of iterations and accuracy vs number of iterations graphs.Also we tested the code by making it predict a digit, randomly chosen from google.
 
 Finally, with the help of opencv we also did the recognition task using:
 1) Mouse as a paint brush
 2) Real time 
           
