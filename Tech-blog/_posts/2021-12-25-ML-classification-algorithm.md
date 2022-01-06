---
layout: post
title:  "ML: Classification Algorithm"
date:   2021-12-25 
category: tech
---

#### Classifiction algorithms are part of supervised learnign algorithms.

**Definition:**
Classification algorithm is a supervised machine learning technique that is used to identify the category of new observation on the basis of training data.

In classification, a program learns from thegiven dataset or observation and then classifies new observation into a number of classes or groups, such as `Yes or No`, `0 or 1`, `Spam or Not spam`, `Cat or Dog`, etc.

Classes  can be called as targets/labels or categores. Unlike regression, the output variable of classification is a category, not a value, such as `Green or Blue`.

Since the classification algorithm is a supervised learning techniqu, hence it takes labeled input data, which means it contains input with the corresponding output.

In classification algorithm, a discrete output function(y) is mapped to input variable(X). 

 y=f(x),where y =categorical output

- The best example of an ML classification algorithm is Email spam detector.
- The main goal of the classification algorithm is to identifty the category of a given dataset, and these algorithms are mainly used to predict the output for the categorical data.


Classification algorithms can be better understood using the diagram below. There are two classes, `Class A` and `Class B`.



















These classes have features that are similar to each other and dissimilar to other classes.

The algorithm which implements the classification on a dataset is known as a classifier.

#### There are two types of classifications:

1. **Binary Classifier:** If the classification problem has only two possible outcomes, then it is called Binary calssifier.
    **Example:** `YES or NO`, `MALE or FEMALE`, `SPAM or NOT SPAM`, etc.      
2.  **Multi-class classifier:** If a classification problem has more than two outcomes, then it is called a Multi-class classifier.
    **Example:** Classification of types of crops & classification of types of music



#### Learners in classification problem

In clasification problem, there are two types of learners:

1. **Lazy learners:** Lazy learners firstly stores the the training dataset and waits until it receives the test dataset.
- In lazy learner case, classification is done on the basis of the most related data stored in the training dataset. It takes less time in training but more time for prediction.

**Example:** K-NN algorith, case-based reasoning.

2. **Eager learners:** Eager learners develop a classification model based on a training dataset before receiving test dataset.

Opposite to lazy learners, eager learner takes more time in learning, and less time in prediction.

**Example:** `Decision trees`, `Naive bayes` & `ANN`





#### Types if ML classification Algorithms:
Classifcation algorithms can be further divided into the mainly two categories:

Linear Models
- Logistic regression
- Support vector machines

Non-linear models
- K-Nearest neighbours
- Kernel SVM
- Naive Bayes
- Decision trees classification
- Random forest classification
- Convolutional Neural Networks (CNN)





#### Evaluating a classification model
The following are the ways evaluating a classification  model, we have :
1. `Log loss` or `cross-entropy loss`
2. `Confusion matrix`
3. `AUC-ROC curve`


**Log loss or cross-entropy loss**
It is used for evaluating the performance of a classifier, whose output is a probability value betwen the 0 and 1

For a good binary classification model, the value of log loss should be near 0.

The value of log loss increases if the predicted value deviates from the actual value The lower the log loss represents the higher value of the model. 

For Binary classifiation, cross-entropy can be calculated as: 




















**Confusion matrix:**
The confusion matrix provides us a matrix /table as output and describes the performance of a model.

It is also known as the error matrix.

The matrix consists of prediction results in a summarized form, which has a total number of correct predictions and incorrect predictions.



**AUC-ROC curve**
`ROC` curve stands for `Receiver Operating Characteristics Curve` and `AUC` stands for `Area Under Curve`. 

It is a graph that shows the performance of the classification model at different threshold.

To visualize the performance of the multi-class classification model, we use the `AUC-ROC` Curve.

The `ROC` curve is plotted with `TPR` and `FPR`, where TPR(True Positive Rate) on Y-axis and FPR(False Positive Rate) on X-Axis.



#### Use cases of Clasification Algorithms.
Classification algorithms can be used diferent place. Below are some popular use cases of classification algorithms:

1. Email spam detection.
2. Speech Recognition.
3. Identification of cancer tumor cells
4. Drugs classification
5. Biometrics identification. etc.


