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























- The task of predicting the class labels of test records can also be concidered as binomial experiment, which can be understood as follows. Let us concider the following:
 N = Number of records in the test set.
 n = Number of records predicted correctly by the classifier.
 E = n/N, The observed accuracy(it is also called the emperical accuracy).
 E = the accuracy



















A table of 






















NOTE:

Suppose, a classifier is tested k times with k different test sets. If E denotes the predicted accuracy when tested with the set  N in the i-th run 

then the overall predicted accuracy is









### Performance estimation
#### Performamce estimation of a classifier

Predictive accuracy works fine, when the classes are balanced 
that is every class in the dataset are equally important

In fact, data set with imbalanced class distributionare quite coommon in many real life applications.

When the classifier classified a test data set with imbalanced class didtributions then, predictive accuracy on its own is not a reliable indicator of a clssifier's effectiveness.



#### Effectiveness of a prrdictive accuracy

Example

Given a data set of stock markets, we are to classify them as `good` and `worst`
Suppose in the data set, out of 100 entries, 98 belong to `good` class and only 2 are in `worst` class

with the data set if the classifier's predictive accuracy is `0.98`, a very high value !

Here there is a high chance that 2 worst  stock market may incorrectly be classified as good 

On the other hand, if the predictive accuracy of  `0.02` then none of the stock markets may be classified as `good`

Thus, when the classifir classified a test data set with imbalanced class distributions, then predictive accuracy on its own is not a reliable indictor of a classifier's effectiveness.

This neccesitates an alternative metrics to judge the classifier.

Before exploring them, we introduce the concept of `Confusion matrix`


#### Confusiom matrix

Example

A classifier is built on a dataset regarding Goog and worst classes of stock markets. The model is then tested with a test set of 10000 unseen instances. The result is shown in the form of a confusion matrix. The result is self explanatory.















Having `m` classes, confusion matrix is a table of size `m * m` where element at (i , j) indicates the number of instances of class `i` but classified as class `j` 

To have good accuracy for a classifier, ideally most diagonal entries should have large values with the rest of entries being close to zero.

Confusion matrix may have additional rows or columns to provide total or recognition rates per class.


> Example: confusion matrix with multiple class

Following table shows the confusion matrix of a classification problem with six classes labeled as 

















In case of multiclass classification, sometimes one class is important enough to be regarded as positive with all other classes combined togather as negative.
Thus a large confusion matrix of `m * m` can be concised into `2 * 2` matrix.

For example, the CM shown in example  11.5 is transformed into a CM of size 2 * 2 considering the class C 1 as the positive class and classes C2, C3, C4, C5, & C6
combined togather as negative 














How can we calculate the predictive accuracy of the classifier model in this case?

Are the predictive accracy same in both examples

We now define a number of metrics for the measurement of a classifier .

In our discussion, we shall make the assumption that there are only two classes: `+ (positive)` & `-(negative)` 

Nevertheless, the metrics can easily be extended to multi-class classifiers(with some modifications)

-`True Positive Rate` (TPR): It is defined as a fraction of thr positive examples predicted correctly by the classifier










This metrics is also known as **Recall, Sensitivity or Hit rate.**

- `False Positive Rate`(FPR): It is defined as the fraction of negative examples classified as positive class by the classifier.








This metric is also known as **False Alarm Rate**


- False Negative Rate (FNR): It is defined as the fraction of positive examples classified as negative class by the classifier.





- True Negative Rate (TNR): It is defined as the fraction of negative examples classified correctly by the classifier






This metric is also known as `specificity`

- Positive predictive value(PPV) It is defined as the fraction of the positive examples as positive that are rally positive 






It is also known as `precision`

F1 scores(F1): Recall(r) and precision (p) are two widely used metrics employed in analysis, where detection of one of the classes is concidered more significant than others.]

It is defined inn terms of (r or TPR) and (p or PPV) as follows











F1 represents the harmonic mean between recall and precision
High value of F1 score ensures that both precision and recall are reasonably high

More generally, `F(beta)` score can be used to determine the trade-off between `Recall` and `precision` as







Both, precision and Recall are special cases of  `F(beta)` when `beta` = 0 and `beta`= 1 respectively







A more generall metric that captures Recall, precision as well as is defined in the following








Note
In fact, given TPR, FPR, p and r, we can derive all others measures, that is, are the universal metrics

It is defined as the fraction to the total number of examples that are correctly classified by the classifier to the total number of instnces 










The accuracy is equivalent to F(w) with W1=w2=w3=w4=1



#### Error rate
The error rate is defined as the fraction of the examples that are incorrectly classified 












predictive accuracy can be expressed in terms of sensitivity and specificity
We can write:











Thus







#### Analysis with perfomance measurement metrics
- Based on the various performance metrics, we can characterize a classifier.
We do it in terms of TPR, FPR, Precision and recall and accuracy


**Case 1: Perfect classifier**

When every instance is correctly classified, it is called the perfect classifier. In this case, TP = P, TN = N and CM is 









**Case 2: Worst classifier**

Whem every instance is wrongly classified, it is called the worst classifier. In this case 











**Case 3: Ultra-Liberal classifier**

The classifier always predicts the + class correctly. Here, the false negative (FN) and the True negative (TN) are zero. The CM is












**Case 4: Ultra-Consevative Classifier**

This classifier always predicts the - class correctly. Here, the false Negative (FN) and True Negative (TN) are zero. The CM is















### Predictive accuracy vs TPR & FPR
- One strength of characterizing a classifier by its TPR and FPR is that they do not depend on the relativ size of P and N. The same is also applicable for FNR and TNR  and other measures from CM

- In contrast, the predictive accuracy, precision, error rate, F1 scores,











