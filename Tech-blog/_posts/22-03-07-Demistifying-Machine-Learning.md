---
layout: post
title:  "AI, ML, & DL :Demystifying Machine Learning"
date:   2022-03-07 
category: tech
---



**Supervised Learning**

Machine learning models aretrained in supervised learning using labeled examples or data. The model is provided with the ground truthfor each examples from which it will learn. In other words, each example consists of the dta and the label that tells the model what the correct output for the data is.
Supervised learning aims to learn from the examples and generalizes its learning to data it hasn't seen before. A model performs well only when you provide **deverse and well-labeled data** for training.


Let’s understand how you can use Supervised Learning to train a model to categorize fruits into apples or oranges. 


#### Step 01
Provide numerical input data representing fruits and their labels. 

#### Step 02
Train the model using a machine learning algorithm to learn a line that can separate the two types of fruit.

#### Step 03
Have the model classify a fruit that it has not seen before as an apple or an orange by predicting what side of the line it will be.


#### Model performance
The model will initially, and frequently be wrong but will use the labels provided during training to correct itself and arrive at the right answer. The model will also perform only as well as the data used to train it. For instance, if you train it using only green apples as examples, the model will only recognize green apples as apples and may struggle with red or yellow ones. The more diverse and labeled your data, the better your model will perform!



**Unsupervised Learning**
In unsupervised learning, the example data provided to the model is unlabeled. You may roughly know the number of classes you expect to discover from the data. The model attempts to find related data and clusters them together. Unsupervised learning is especially useful when dealing with datasets that are difficult to label. An everyday use case for unsupervised learning is recommendation engines commonly found in retail websites, where the system clusters people with similar shopping habits to recommend new products. 


#### Reinforcement learning

Reinforcement learning is a relatively new area of machine learning where the model takes actions to achieve a goal while maximizing a reward that is defined. This system relies on trial and error with accurate “trials” providing rewards. The system undergoes many iterations to find a combination of rules that achieves the best results. The applications of reinforcement learning include gaming, robotics, and scientific research.