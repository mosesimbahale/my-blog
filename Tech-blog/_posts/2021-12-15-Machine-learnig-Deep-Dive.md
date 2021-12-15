---
layout: post
title:  "Statistical Learning Theory"
date:   2021-12-15 
category: tech
---

# Probably Approximately Correct (PAC) Learning

**PAC** Learning is a framework for the mathematical analysis of machine learning

Goal of PAC : With higj probability("probably"), the selected hypothesis will have lower error(Approximately correct)

In the PAC model, we specify two small parameters, Exponential and Omega, and require that with probability at least (1 - omega) a system learn a concept with a error at exponential.
Exponential givesan upper bound on the error in accuracy with which h approximated (accuracy: 1 -Exponential)
Omega gives the probability of failure in achieving this accuracy(confidence: 1-Omega)


**Example:** Learn the concept "Medium-built person"

- We are given the height and weight of m individuals, the train set.
- We are told for each[height, weight] pair if the person is medium buil or not.
- We would like to learni this concept, i.e produce an algorithm that in the future correctly answers if a pair[height, weight] represents a medium-built person or not.

![image](https://user-images.githubusercontent.com/42868535/146161458-a468808b-e2ee-4b7c-b944-7469f8743db1.png)


![image](https://user-images.githubusercontent.com/42868535/146161540-8605a397-5de3-494f-ba38-91f6608ad870.png)

False Negative False Negative is the region that is classified by hypothesis C as Negative while hypothesis h classifies it as postive

![image](https://user-images.githubusercontent.com/42868535/146161667-de9d1e84-b6fd-458b-8668-b45131e3d49b.png)

False Postive is the region that is classified by hypothesis C as positive while hypothesis h classifies it as negative

![image](https://user-images.githubusercontent.com/42868535/146161791-de7a7b30-d2ac-4ea3-8087-df9e4bc6f4ad.png)

**XOR** also pronounced as Exclusive OR) gives a true (1 or HIGH) output when the number of true inputs is odd. That is, if one, and only one, of the inputs to the gate is true. If both inputs are false (0/LOW) or both are true, a false output results.

![image](https://user-images.githubusercontent.com/42868535/146161959-cc6b8ccc-97e9-47e7-8cca-40fd8b99d3a5.png)

![image](https://user-images.githubusercontent.com/42868535/146162034-156060cf-d2f3-440e-9c07-d196f2dd1f85.png)

Since there is no learner that can learn a concept with 100% accuracy, therefore we need a hypothesis that is approximately correct. Therefore, our aim is to reduce the probability of the error region ![image](https://user-images.githubusercontent.com/42868535/146162124-d5c7330b-0dc5-4548-b0a0-9a32b904766a.png) Therefore a hypothesis is said to be approximately correct, if the error is less than or equal to Exponential, where ![image](https://user-images.githubusercontent.com/42868535/146162304-d9c1cc6b-7990-4d67-a938-58322f9d7f72.png)


## Probably Approximately Correct (PAC) Learning 

- Since the training samples are drawn randomly, there is always a probability that a training samples encountered by a learner will be misleading.
- If the samples drawn are not the actual representation of the real instances, the hypothesis generated may not be approximately correct.
- Therefore for a specific learning algorithm, what is the probability that the concept it learns will have an error that is bounded by ε and whose failure probability is bounded by ![image](https://user-images.githubusercontent.com/42868535/146162746-a085f24e-3a23-4456-aa17-edb61dc3177a.png)

- That is ![image](https://user-images.githubusercontent.com/42868535/146162800-0b567489-8ebe-4b0a-8f30-b5b80e862bb3.png)
- Hence the goal is to achieve low generalization error with high probability.

![image](https://user-images.githubusercontent.com/42868535/146162935-2892b81c-877d-4400-8939-07aabfd3507b.png)

![image](https://user-images.githubusercontent.com/42868535/146163026-b9cc07ff-dffb-4e75-8fd1-8e2c531138f6.png)

![image](https://user-images.githubusercontent.com/42868535/146163124-f6bff384-6f31-4216-86de-29ae4642ccd4.png)

![image](https://user-images.githubusercontent.com/42868535/146163216-b4f1a0a8-559c-491e-b026-45ccbd567f7a.png)

![image](https://user-images.githubusercontent.com/42868535/146163381-1b813f93-c58f-4ec5-a7c9-5ab5a689087b.png)



# PAC-learnability axis-aligned rectangles 

**Consider an hypothesis to be an axis-aligned rectangles**

Assume C is the real target function. Our goal is to find the best rectangle h that approximates the realtargetfictionC Now hs is the tightest possible rectangle around a set of positive trainin gexamples. 

![image](https://user-images.githubusercontent.com/42868535/146164096-0c3479fc-f937-45d0-a86f-ff375778ecf4.png)

![image](https://user-images.githubusercontent.com/42868535/146164160-d79d9d17-b3fd-4e6b-9015-9224bc3cb17b.png)

Weneedtoshowthatthegeneralizationerrorishighwithprobability.The shaded region represents the error region and I should be bounded by ε. If a hypothesis lies within this shaded region it is deemed to be approximately correct.

![image](https://user-images.githubusercontent.com/42868535/146164254-54862ff6-03b3-4566-aeaa-3f060e118018.png)

`Approximately correct hypothesis`

![image](https://user-images.githubusercontent.com/42868535/146164346-2fb38dfd-5bb1-4690-9d50-4d171a666f4a.png)

`The hypothesis error is greater than ε`

![image](https://user-images.githubusercontent.com/42868535/146164483-f070d161-b0fb-4a69-915c-168710fa738b.png)

![image](https://user-images.githubusercontent.com/42868535/146164532-280150a6-2d1d-47f5-8fa5-3e626aa801f0.png)

![image](https://user-images.githubusercontent.com/42868535/146164558-61e224ff-c574-499c-b124-2bc133a4c0ec.png)

# Conclusion 

- PAC learning is used in classification to learn  how we can put a bound on a true error given training error. 
- In other words, how we can reduce generalizabilityerror but still achieving a high probability. 

