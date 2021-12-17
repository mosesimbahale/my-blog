---
layout: post
title:  "Tensorflow: Image processing with machine learning"
date:   2021-12-15 
category: tech
---

# Image processing with machine learning.

**A summary on Basics of TensorFlow Tutorial Using Fashion MNIST**

Machine learning (ML)/Neural Network (NN) tools have recently made a huge splash with applications in data analysis, image classification, and data generation. Although ML methods have existed for decades, recent advancements in hardware have generated systems powerful enough to run these algorithms.

The typical "hello world" example for ML is a classifier trained over the MNIST dataset; a dataset of the handwritten digits 0-9. This dataset is getting a little stale and is no longer impressive with employers as a proof of capability due to both its seeming simplicity and to the plethora of existing tutorials on the topic. Here we will use a newer dataset to perform our ML "hello world", the Fashion MNIST dataset!

The Zeroth Step of ML (that should be completed before ever putting a hand to mouse, or finger to key) is understanding the format and sizes of your data. This step is often referred to as feature engineering. Feature engineering, typically, includes selecting and preprocessing the particular aspects of training data to give to your algorithm. 

The Fashion MNIST dataset is comprised of 70,000 grayscale images of articles of clothing. The greyscale values for a pixel range from 0-255 (black to white). Each low-resolution image is 28x28 pixels and is of exactly one clothing item. Alongside each image is a label that places the article within a category; these categories are shown in Figure 2 with an example image belonging to the class.

## Gettng started

STEP O1: INSTALL PYTHON:
As a first step, let's make sure Python is installed and running. To test if it is installed and configured already, type python into your terminal. If it isn't installed yet, it should say something like "python is a unknown command". If it is installed, it will open the python environment, and should look something like this:

```
Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>

```
If you see this type exit() to return to the command prompt. If not, you need install the latest Python 3 release for your OS: Windows, Mac, Linux/Unix. After you install python, you need to add it to your PATH variable to use the python command shortcut. You can find directions for how to do this here

You will need to close and reopen you command prompt before the new environmental variable is recognized. Type python to check if it is set up correctly. You will also know that you are in the Python environment as the prompt at which you type is represented by ">>>". Now, to install some packages for our project, we cannot do this from the Python environment, so we will need the means to leave the environment. This command is conveniently exit().

To complete this project we will need a few packages; these are add-ons available to Python but not included in the base install. Very luckily, we installed the latest Python 3, which includes the Pip Installation module of Python. Pip is a fast and easy way to install packages and their dependencies. As we are doing an NN-based project, we will need to use TensorFlow. For those who only recognize TensorFlow by its association with NNs, it may be shocking to learn that TensorFlow is a more general tool for the manipulation of mathematical entities called tensors. NNs are just a single use of tenor mechanics, and therefore TensorFlow.

As tensors and TensorFlow can be fairly complex to manage, there exists another package named Keras that acts as a high-level API (Application Programming Interface), allowing users to easily generation, define and manipulate the structures within TensorFlow

Finally, as we wish to visualize aspects of our dataset and generate some informational plots, we will want Python's Mathematical Plotting Library, MatPlotLib. Additionally, MatPlotLib facilitates the generation of graphical plots in new windows, even when executed from the Command Line. And finally, to handle numerical computations and array-based operations we will import Numpy.

Starting from the c-prompt (where we left off above), you will need to tell the easy Python package manager (a program that helps you install, uninstall, update and upgrade ancillary features to a larger application) that we want to install Numpy, MatPlotLib, and Tensorflow.


```
pip install Numpy
pip install MatPlotLib
pip install Tensorflow
```

If you encounter a permissions error, you may need to add `--user` to the end of each install. For example `pip install Numpy --user.`

After each of these actions, pip will display that it is downloading and installing the package. If you have an existing (but not up-to-date) version, pip will first uninstall the old version before installing the new. If you have previously installed the current version of any package, pip will inform you that the 'requirement already exists'.


Now we can test if Python has installed our Tensorflow correctly (this being the testier of the three packages), by returning to the Python Environment and printing the version of our Tensorflow:

```
python
Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
1.14.0
>>> exit()
```
We can see that I have Tensorflow version 1.14.0 installed! Unless you have specified an earlier version (and dialing in the version that you need for any particular existing project can be a hassle with Tensorflow) you will have automatically installed the newest stable release, which at the time of this writing was version 2.0.0.
And yes, I have intentionally had to enter and exit the Python environment from the CMD several times. This is called Reinforcement Learning; you've been subjected to the human equivalent of another machine learning technique you will learn about later.

TensorFlow is only used for Neural Networks: TensorFlow is a more general tool for working with tensors. Therefore, it is not only used for Neural Networks.



## **Loading dataset:**

Now we are ready to roll! First, we must admit that it takes a lot of data to train a NN, and 70,000 examples is an anemic dataset. So instead of doing a more traditional 70/20/10 or 80/10/10 percent split between training/validating/testing, we will do a simple 6:1 ratio of training:testing (note that this is not best practices, but when there is limited data it may be your only recourse).


```
>>> fashion_mnist = keras.datasets.fashion_mnist
>>> (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 1us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 2s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
>>>
```


The first line merely assigns the name fashion_mnist to a particular dataset located within Keras' dataset library. The second line defines four arrays containing the training and testing data, cleaved again into separate structures for images and labels, and then loads all of that data into our standup of Python. The training data arrays will be used to --you guessed it-- train the model, and the testing arrays will allow us to evaluate the performance of our model.


# VISUALIZE THE DATA

It's always nice to be able to show that we've actually done something; ever since kindergarten there has been no better way than with a picture! You'll note that we pip installed and imported MatPlotLib, a library for plots and graphs. Here we'll use it to visualize an example of the Fashion MNIST dataset.

```
>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> plt.imshow(train_images[0])
<matplotlib.image.AxesImage object at 0x00000133F2152518>
>>> plt.colorbar()
<matplotlib.colorbar.Colorbar object at 0x00000133F2184A90>
>>> plt.grid(False)
>>> plt.show()
```


The first command basically generates a figure object that will be manipulated by commands 2 through 4. Command 2 specifies what it is that we shall be plotting: the first element from the train_images array. NOTE: Recall that python is an inclusive counting language, meaning that it numbers/indexes things starting from zero, not one! And the final command, "show()", tells Python to generate this figure in an external (from CMD) window.



![image](https://user-images.githubusercontent.com/42868535/146585148-db1a34c5-84c0-43b4-90e2-223f25001913.png)

> Figure 3 The graphical output of the above code snip. Note that this is a pixelated ankle boot image in greyscale that has been false-colored.



Your window should contain a plot that looks similar to Figure 3. Also, be aware that after plt.show(), Python will not return you to a command line until the newly generated window (containing our super nice picture) is closed. Upon closing the window, you will be able to continue entering Python commands.


**What a fine looking boot! 👢**

# Lets write a program that will classify this boot
