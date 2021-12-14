---
layout: post
title:  "Machine Learning : Decision trees"
date:   2021-12-14 10:30:35 
category: tech
---

# Decision Trees

Decision Trees are a type of Supervised Machine learning where the data is continuously split according to a certain parameter and are for classification and regression. Generally, they learn a hierarchy of if/else questions, leading to a decision. The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes while the decision nodes are where the data is split.  Basically, a decision tree is a flowchart-like tree structure, where  


- Each internal node denotes a test on an attribute.  
- Each branch represents an outcome of the test. 
- Each leaf node holds a class label.  
- The topmost node in a tree is the root node.  

**Example** 

 Consider the following training dataset: 


![image](https://user-images.githubusercontent.com/42868535/146062326-4a3ed2dc-76e0-454f-883f-117f4936e5f8.png)


`Output: A Decision Tree for “buys_computer" `

![image](https://user-images.githubusercontent.com/42868535/146062438-bd58fc63-718a-4e71-8983-520cba3fce6d.png)


The decision nodes are questions like ‘What’s the age?’, ‘Is he a student?’, and ‘What’s is his credit ratings?’ And the leaves, which are outcomes like either ‘yes’, or ‘no’. 

There are two main types of Decision Trees:  

**Classification trees** (Yes/No types)  
The outcome was a decision variable like Categorical e.g. from the above example is ‘yes’ or ‘no’. 

**Regression trees** (Continuous data types) 
The decision or the outcome variable is Continuous, e.g. a number like 123. 

**Building decision trees**
Learning a decision tree is learning the sequence of if/else questions. In the machine learning setting, these questions are called tests (not to be confused with the test set). The data is usually represented as continuous features such as in the 2D dataset shown in the figure below. The tests that are used on continuous data are of the form “Is feature i larger than value a?” 


![image](https://user-images.githubusercontent.com/42868535/146062967-648fff9e-86e5-490d-9bdb-8f676d9ef717.png)

the form “Is feature i larger than value a?” To build a tree, the algorithm searches over all possible tests and finds the one that is most informative about the target variable. The diagram below shows the first test that is picked. Splitting the dataset vertically at x[1]=0.0596 yields the most information; it best separates the points in class one from the points in class two. The top node (root node), represents the whole dataset. The split is done by testing whether x[1] <= 0.0596, indicated by a black line. If the test is true, a point is assigned to the left node, which contains 2 points belonging to class 0 and 32 points belonging to class 1. Otherwise the point is assigned to the right node, which contains 48 points belonging to class 0 and 18 points belonging to class 1. These two nodes correspond to the top and bottom regions shown in the diagram. Even though the first split did a good job of separating the two classes, the bottom region still contains points belonging to class 0, and the top region still contains points belonging to class 1. 


![image](https://user-images.githubusercontent.com/42868535/146063144-086edd7e-3dd7-41fd-b2ea-38026598b1aa.png)

A more accurate model can be built by repeating the process of looking for the best test in both regions. The following diagram shows that the most informative next split for the left and the right region is based on x[0].

![image](https://user-images.githubusercontent.com/42868535/146063200-3b4f4236-31cf-4e9f-b250-36115dae93a2.png)

This recursive process yields a binary tree of decisions, with each node containing a test.  

The recursive partitioning of the data is repeated until each region in the partition (each leaf in the decision tree) only contains a single target value (a single class or a single regression value). A leaf of the tree that contains data points that all share the same target value is called pure. The final partitioning for this dataset is shown in below 

![image](https://user-images.githubusercontent.com/42868535/146063430-320633d6-80df-4a1e-9465-42efafba1286.png)

A prediction on a new data point is made by checking which region of the partition of the feature space the point lies in, and then predicting the majority target (or the single target in the case of pure leaves) in that region. The region can be found by traversing the tree from the root and going left or right, depending on whether the test is fulfilled or not. 

Generally, in machine learning implementations of decision trees, the questions generally take the form of axis-aligned splits in the data i.e. each node in the tree splits the data into two groups using a cutoff value within one of the features.


It is also possible to use trees for regression tasks, using exactly the same technique. To make a prediction, we traverse the tree based on the tests in each node and find the leaf the new data point falls into. The output for this data point is the mean target of the training points in this leaf. 


**Controlling complexity of decision trees**
Building a tree as described above and continuing until all leaves are pure leads to models that are very complex and highly overfit to the training data i.e. the presence of pure leaves mean that a tree is 100% accurate on the training set; each data point in the training set is in a leaf that has the correct majority class. The overfitting can be seen on the left of the above diagram. You can see the regions determined to belong to class 1 in the middle of all the points belonging to class 0. On the other hand, there is a small strip predicted as class 0 around the point belonging to class 0 to the very right. 


**There are two common strategies to prevent overfitting:**
- Stopping the creation of the tree early (also called pre-pruning) 
- Building the tree but then removing or collapsing nodes that contain little information (also called post-pruning or just pruning). 

Possible criteria for pre-pruning include limiting the maximum depth of the tree, limiting the maximum number of leaves, or requiring a minimum number of points in a node to keep splitting it. 


**Decision trees in scikit-learn**
Decision trees in scikit-learn are implemented in the DecisionTreeRegressor and DecisionTreeClassifier classes. scikit-learn only implements pre-pruning. 

**Example**
Consider the following two-dimensional data, which has one of four class labels 

```
In[2]: import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow'); 

```

![image](https://user-images.githubusercontent.com/42868535/146064160-988ab8c1-4c2b-403a-9608-2d88e39b555a.png)

**Example**
In the following example, you are required to implement DecisionTreeclassifier estimator on Wisconsin Breast Cancer dataset and create a model to predict whether a tumor is malignant based on the measurements of the tissue. Notice that scikit-learn includes two real-world datasets. One is the Wisconsin Breast Cancer dataset (cancer, for short), which records clinical measurements of breast cancer tumors. Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors).  


