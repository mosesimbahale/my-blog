---
layout: post
title:  "Tensorflow: Image processing with machine learning"
date:   2021-12-15 
category: tech
---

# Introduction: Tensorflow Image processing.

**A summary on Basics of TensorFlow Tutorial Using Fashion MNIST**

Machine learning (ML)/Neural Network (NN) tools have recently made a huge splash with applications in data analysis, image classification, and data generation. Although ML methods have existed for decades, recent advancements in hardware have generated systems powerful enough to run these algorithms.

The typical "hello world" example for ML is a classifier trained over the MNIST dataset; a dataset of the handwritten digits 0-9. This dataset is getting a little stale and is no longer impressive with employers as a proof of capability due to both its seeming simplicity and to the plethora of existing tutorials on the topic. Here we will use a newer dataset to perform our ML "hello world", the Fashion MNIST dataset!

The Zeroth Step of ML (that should be completed before ever putting a hand to mouse, or finger to key) is understanding the format and sizes of your data. This step is often referred to as feature engineering. Feature engineering, typically, includes selecting and preprocessing the particular aspects of training data to give to your algorithm. 

The Fashion MNIST dataset is comprised of 70,000 grayscale images of articles of clothing. The greyscale values for a pixel range from 0-255 (black to white). Each low-resolution image is 28x28 pixels and is of exactly one clothing item. Alongside each image is a label that places the article within a category; these categories are shown in Figure 2 with an example image belonging to the class.

## Gettng started

**STEP O1: INSTALL PYTHON:**
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



**STEP O2:**
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

**STEP O3:** 
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

# Lets write a program that will classify this boot as a boot!



**STEP O4:** 
**Preprocesing the dataset**
The greyscale assigned to each pixel within an image has a value range of 0-255. We will want to flatten (smoosh… scale…) this range to 0-1. To achieve this flattening, we will exploit the data structure that our images are stored in, arrays. You see, each image is stored as a 2-dimensional array where each numerical value in the array is the greyscale code of particular pixel. Conveniently, if we divide an entire array by a scalar we generate a new array whose elements are the original elements divided by the scalar.

```
>>> train_images = train_images / 255.0
>>> test_images = test_images / 255.0
>>>
```

Two vital notes about the above.

1. Use the value "255.0". This value is a floating point number (float), and will always return a float during algebraic operations. In Python, the division operator always returns a float to avoid rounding; but, that is not true for all programming languages, so it's a good habit to include that decimal because it automatically sets that number to be a float.

2. Do not rescale the train_labels or test_labels arrays, these values are already in the range 0-9, as they should be!

**Remember, the label arrays are only used to associate images with their lables.**


**STEP O5:** 
# **Model generation**
Every NN is constructed from a series of connected layers that are full of connection nodes. Simple mathematical operations are undertaken at each node in each layer, yet through the volume of connections and operations, these ML models can perform impressive and complex tasks.

Our model will be constructed from 3 layers. The first layer – often referred to as the Input Layer – will intake an image and format the data structure in a method acceptable for the subsequent layers. In our case, this first layer will be a Flatten layer that intakes a multi-dimensional array and produces an array of a single dimension, this places all the pixel data on an equal depth during input. Both of the next layers will be simple fully connected layers, referred to as Dense layers, with 128 and 10 nodes respectively. These fully connected layers are the simplest layer in the sense of understanding, yet allow for the greatest number of layer-to-layer connections and relationships.

The final bit of hyper-technical knowledge you'll need to learn is that each layer can have its own particular mathematical operation. These activation functions determine the form and relationship between the information provided by the layer. The first dense layer will feature a Rectified Linear Unit (ReLU) Activation Function that outputs values between zero and 1; mathematically, the activation function behaves like f(x)=max(0,x). The final layer uses the softmax activation function. This function also produces values in the 0-1 range, BUT generates these values such that the sum of the outputs will be 1! This makes the softmax a layer that is excellent at outputting probabilities.


```
>>> model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

WARNING: Logging before flag parsing goes to stderr.
W0824 22:50:02.551490  8392 deprecation.py:506] From C:\Users\ross.hoehn\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
>>>
```

**Softmax activation not only flattens each value (between 0 and 1) but also scales everything to add up to 1.**


**STEP O6:** 
# Training the model:
Models must be both compiled and trained prior to use. When compiling we must define a few more parameters that control how models are updated during training (optimizer), how the model's accuracy is measured during training (loss function), and what is to be measured to determine the model's accuracy (metrics). These values were selected for this project, yet are generally dependent on the model's intent and expected input and output.


``` 
>>> model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
>>>
```

Now we can begin training our model! Now, with already having generated and compiled the model, the code required to train the model is a single line.

```
>>> model.fit(train_images, train_labels, epochs=5)

2019-08-24 22:56:32.884249: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Epoch 1/5
60000/60000 [==============================] - 2s 40us/sample - loss: 0.4985 - acc: 0.8264
Epoch 2/5
60000/60000 [==============================] - 2s 36us/sample - loss: 0.3787 - acc: 0.8632
Epoch 3/5
60000/60000 [==============================] - 2s 36us/sample - loss: 0.3368 - acc: 0.8766
Epoch 4/5
60000/60000 [==============================] - 2s 35us/sample - loss: 0.3122 - acc: 0.8863
Epoch 5/5
60000/60000 [==============================] - 2s 35us/sample - loss: 0.2962 - acc: 0.8901
<tensorflow.python.keras.callbacks.History object at 0x00000133F219C470>
>>>
```

This single line completes the entire job of training our model, but let's take a brief look at the arguments provided to the model.fit command.

1. The first argument is input data, and recall that our input Flatten layer takes a (28,28) array, conforming to the dimensionality of our images.

2. Next we train the system by providing the correct classification for all the training examples.

3. The final argument is the number of epochs undertaken during training; each epoch is a training cycle over all the training data. Our setting the epoch value to 5 means that the model will be trained overall 60,000 training examples 5 times. After each epoch, we get both the value of the loss function and the model's accuracy (88.97% after epoch 5) at this epoch.


> NOTE: The second argument in the model.fit method is used to classify our data into categories



**STEP O7:** 
# **Evaluating Our Model**
***Recap:***
Now we are working with a functional and trained NN model. Following our logic from the top, we have built a NN that intakes a (28,28) array, flattens the data into a (784) array, compiled and trained 2 dense layers, and the softmax activation function of the final output layer will provide a probability that the image belongs to each of the 10 label categories.

Our model can be evaluated by using the model.evaluate command, that takes in the images and labels so that it can compare its predictions to the ground truth provided by the labels. Model.evaluate provides two outputs, the value of the loss function over the testing examples, and the accuracy of the model over this testing population. The important output for us is the model's accuracy.

```
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
10000/10000 [==============================] - 0s 27us/sample - loss: 0.3543 - acc: 0.8721
>>> print(test_acc)
0.8721
>>>
```

This is great! Our model performs at an accuracy of 87.21%. As good as that is, it is lower than the model accuracy promised above (89.01%). This lower performance is due to the model overfitting on the training data. Overfitting occurs when there are too many parameters within the model when compared to the number of training instances; this allows the model to over learn on those limited examples. Overfitting leads to better model performance over non-training data.

That said, 87.21% is a decent number! Let's finally learn how you can feed our model the series of test examples from the test_images array, and have it provide its predictions.

```
>>> predictions = model.predict(test_images)
>>> predictions[0]
array([5.1039719e-04, 1.4324225e-07, 6.3209918e-06, 1.4587535e-07,
       7.1591121e-06, 3.9024312e-02, 3.2491367e-05, 9.4579764e-02,
       1.8918892e-05, 8.6582035e-01], dtype=float32)
```


As we can see, most of the entries in our prediction array are very close to 0. They are written in scientific notation--the value after the e being the number decimal places to adjust the value (for example 5.1 e-04 is actually 0.00051). The entry that stands out is predictions[0][9] at .8658, or 86.58%, certainty that this image should be classified as a boot!

If you prefer to not look through a list to determine the class label, we can simplify the output by:

```
>>> numpy.argmax(predictions[0])
9
>>>
```

Finally, we can verify this prediction by looking at the label ourselves:

```
>>> test_labels[0]
9
>>>
```

In the prediction array generated by our model: each number represents: The probability that the image matches the corresponding label in our set of labels

# Conclusion:
There you have it! You have built and trained your first neural network from scratch, and properly classified a boot as a boot!

# Next seps Recommended
- Try using the model on a item of clothing outside the dataset (make sure to preprocess it first so it is on the same scale as the other images).

- Find another [image dataset](https://blog.cambridgespark.com/50-free-machine-learning-datasets-image-datasets-241852b03b49) to try this out on.

- Make an interface that responds with a label when you select a clothing image.

A complete source code of this can be found [here](https://github.com/mosesimbahale/ML-Projects/blob/main/Image_processing_Tensorflow%20(1).ipynb) 


