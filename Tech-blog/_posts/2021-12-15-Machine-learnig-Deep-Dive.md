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
















