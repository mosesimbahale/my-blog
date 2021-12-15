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




