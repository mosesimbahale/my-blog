---
layout: post
title:  "Data analytics: Sensitivity analysis "
date:   2021-12-25 
category: tech
---

#### Outline

- Introduction
  Estimation Strategies
- Accuracy Estimation
- Error Estimation
- Statistical Estimation
- Performance Estimation
- ROC Curve





## Introduction.

A classifier is used to predict an outcome of a test data.

- Such a prediction is useful in many applications for instance Bussiness forecasting, cause-and-effect analysis etc.

A number of classifiers hav been evolved to support the activities and each one of their of their own merits and demerits.

There is need to estimate the accuracy and performance of the classifier with respect to few controlling parameters in data sensitivity.

As a task of sensitivity analysis, we focus on;
- Estimation strategy
- Metrics of measuring accuracy
- Metrics of measuring performance


### Estimatioon strategy
`plannig for estimation`

Using some `trainig data` buildng a classifier based on certain principle is called `learning a classifier`.

After building a classifier and before using it for classifcation of unseen instance, we have to validate it using some `test data`.

Usually training data and test data are outsourced from a large pool of data already available.

`Hold out method`

This is a basic concept of estimating a prediction.
- Given a dataset, it is partitioned into two disjoint sets called `training set` & `testing set`
- Classifier is `learned` based on the training set and get `evaluated` with `testing set`.
- Proportion of training and testing set is the description of analyst; typically `1:1` or `2:1`, and there is a `trade-off-between these sizes` of these two sets.
- if the training set is too large, then model maybe good enough, but wstimation may be less reliable due to small testing and vice-versa.



### Random subsampling
It is a `variation of holdout method` to overcome the `drawback of over-presenting a class` in one set thus under presenting it in other set and vice-versa.

In this method, `Holdout method is repeated k times`, and in each time, `two disjoint sets` are `chosen at random` with a `predefined sizes`

Overall estimation i taken as the average of estimations obtained from each iteration.


### Stratified sampling
When random selecting training or validation set, we ma want to ensure that class proportions are maintained in each selected set. This can bedone through sttratified sampling: First stratify instances by class, then randomly select instances from each class proportionally.


### Cross validation
The main draw backs of random subsampling is, `it does not have contol over the number of times ech tuple is used for tarining and testing`. Cross-validation is proposed to overcome this problem.

There are two main types of cross-validation method.
- K-fold cross-validation
- N-fold cross-validation

### K-fold cross-validation
Dataset consisting of N tuples is devided ito k(usually, `5 or 10`) equal, mutually exclusive parts od folds(D1,D2,....,Dk), and if N is not divisible by k, then te last part will have fewer tuples than other (k-1) parts.

A series of k runs is carried out with this decomposition, and in ith iterations Di is used as test data and other folds as training data. Thus `each tuple is used same number of times for training and ince for testing`

Overall estimates is taken as the average of estimators obtained from eachiteration.
























### N-fold Cross-validation 
In N-fold cross validation method (K-1)/N part is used un training with k-tests.

N-fold cross validation is an `extreme case of K-fold` cross validation, often known as `leave-one-out` cross validation.

Here, dataset is divided into as many folds as there are instances; thus, all most each tuple forming a training set, building N classiifies.

In this method, therefore, N classifiers are built from N-1 instances , and each tuple is used to classify a single test instance

Test sets are mutually exclusive abd effectively cover the entire set(in sequence). This is as if `trained by entire data as well as tested by entire dataset`. Overall estimation is then averaged out of the results of N classifiers.


### N-fold cross-validation: issue
So far the estimators of accuracy and performance of a classifier model is concerned, the `N-fold cross-validation is comparable to the others` we have just discussed.

The drawback of N-fold cross validation strategy is that it is `computationally expensive`, as there is particularly true when data set is large.

In practice, the method is extremely beneficial with very small dataset  only, where as musch data as possible toneed to be used to train  a classifier.





### Bootstrap method.
The bootstrap method is a variation of `repeated version of Random sampling` method. The method suggests the `sampling of training records with replacement`. Each time a record is selected for training set, is put back into the original pool of records, so that it is equally likely to be redrawn in the next run.

In other words, the Bootstrap method samples the givr dataset `uniformly` with `replacements`.

The rationa of having this strategy is that let some records be occur more than once in the samples of both training as well as testig.

> ***What is the probability that a sample will be selected more than once?***
Suppose, we have given a data set of N records. The data set is sampled N-times with replacement, resulting in a bootstrap sample(i.e., training set) of I samples. Note that the entire runs are called a bootstrap sample in this method.

There are certain chances(i.e., probability) that a particular tuple occur one or more times in the training set.
- If they don't appear in training set, they will end up in the test set.
- Each tuple has a probability of being selected






















### Accuracy Estimation
We have learned how a classifier system can be tested. Now we are to learn the metrics with which a classifier should be estimated. There are mainly two things to be measured for a given classifier;
- Accuracy 
- Performance

**Accuracy estimation**























#### Accuracy: True & Predictive
Now, this accuracy may be true(`or absolute`) accuracy or predicted(`or optimistic`) accuracy. True accuracy of a classifier is the accuracy when the classifier is tested with all possible unseen instances in the given classification space.

However, the number of possible unseen instances is potentially very large(`if not infinite`). For example, classifying a hand written character hence, measuring the true accuracy beyond the dispute is impractical.

`Predictive accuracy` of a classifier is an `accuracy estimation for a given test data` (Which are mutually exclusive with the training data)

If the predictive accuracy for test is 














The predictive accuaracy when estimated with a given test set it should be acceptable without any objection.


### Predictive accuracy
> ***Example 1: University of predictive accuracy***

Consider a classifier model 













With the above mentioned issue in mind, researchers have proposed two hueristic measures;
- Error estimation using `Loss functions`
- Statistical Estimating using `Confidence level`



### Error estimation using Loss Functions




























### Statistical estimation using confidence level
In fact, if we know the value of predictive accuracy, say `E`, then we can guess the true accuracy within a certain range given a confidence level.

`confidence level:` The concept of `confidence level` can be better understood with the following two experiments, related to tossing a coin.

***Experiment1:*** When a coin is tossed, there is a probability that the head will occur. We have to experiment the value for this probability value. A simple experiment is that the coin is tossed many times and both numbers of heads and tails are recorded.













Thus, we can say that p -> 0.5 after a large number of trials in each experiment.

***Experiment 2:*** A similar experiment but with different counting is conducted to learn the probability that a coin is flipped its head 20 times out of 50 trials. This experiment is popularly known as `Bernoulli's trials`. It can be tested as follows























