---
layout: post
title:  "Machine Learning :Decision trees Algorithm"
date:   2021-12-14 10:30:35 
category: tech
---

![image](https://user-images.githubusercontent.com/42868535/146654871-89bc0677-f587-412b-af80-aafe2b408fee.png)


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

a) Import the required dataset 

```
In[4]: from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer() 
print("Shape of cancer data: {}".format(cancer.data.shape))

Out[3]: Shape of cancer data: (569, 30) 
```

b) Importing the required libraries for the decision tree analysis 

```
In[5]: from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
```

c) Dividing the data into training and testing sets 

```
In[6]: X_train, X_test, y_train, y_test = train_test_split( 
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
```

d) Building the Decision Tree Model using scikit-learn and determine the accuracy 

```
In[7]:tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, 
y_train))) 
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test))) 

Out[7]: 
Accuracy on training set: 1.000 
Accuracy on test set: 0.937 
```

The training set is 100%. This is because the leaves are pure asd the tree was grown deep enough that it could perfectly memorize all the labels on the training data i.e. if you don’t restrict the depth of a decision tree, the tree can become arbitrarily deep and complex. Unpruned trees are prone to overfitting and not generalizing well to new data.  Apply pre-pruning to the tree, will stop developing the tree before it is perfectly fit to the training data. 
One option is to stop building the tree after a certain depth has been reached. You can set max_depth=4, meaning only four consecutive questions can be asked and this decreases overfitting. This leads to a lower accuracy on the training set, but an improvement on the test set as shown: 

```
In[8]: tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train,
y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

Out[8]:
Accuracy on training set: 0.988 
Accuracy on test set: 0.951 
```

# Analyzing decision trees 

Trees can be visualized by using the export_graphviz function from the tree module. This writes a file in the .dot file format, which is a text file format for storing graphs. You can set an option to color the nodes to reflect the majority class in each node and pass the class and features names so the tree can be properly labeled: 

```
In[9]: from sklearn.tree import export_graphviz 
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
feature_names=cancer.feature_names, impurity=False, 
filled=True) 
```



e) You can read this file and visualize it, as shown in the diagram below, using the graphviz module (or you can use any program that can read .dot files) .i.e. i.e. for plotting the tree, you need to use graphviz and pydotplus. 


```
In[10]: import graphviz
with open("tree.dot") as f: 
dot_graph = f.read() 
graphviz.Source(dot_graph)

```
![image](https://user-images.githubusercontent.com/42868535/146065834-b96a2402-5c92-4380-8676-ce26265d2f5b.png)

The visualization of the tree provides a great in-depth view of how the algorithm makes predictions, and is a good example of a machine learning algorithm that is easily explained to non-experts The n_samples gives the number of samples in that node, while value provides the number of samples per class.


**Feature importance in trees**
Instead of looking at the whole tree, which can be taxing, there are some useful properties that we can derive to summarize the workings of the tree. The most commonly used summary is feature importance, which rates how important each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means “not used at all” and 1 means “perfectly predicts the target.” The feature importance always sum to 1: 

``` 
In[11]:  print("Feature importances:\n{}".format(tree.feature_importances_)) 

```
![image](https://user-images.githubusercontent.com/42868535/146066071-f0c8c43a-970a-4a1a-880b-a4d2edd4fed5.png)

**Feature importance’s can be visualized using the following:**




```
In[12]: import numpy as np 
def plot_feature_importances_cancer(model):
n_features = cancer.data.shape[1] 
plt.barh(range(n_features), model.feature_importances_, 
align='center') 
plt.yticks(np.arange(n_features), cancer.feature_names) 
plt.xlabel("Feature importance") plt.ylabel("Feature") 
plot_feature_importances_cancer(tree)  
```

![image](https://user-images.githubusercontent.com/42868535/146066412-961c3f45-d958-4110-9105-707add0d995d.png)



The feature used in the top split (“worst radius”) is by far the most important feature. This confirms the observation in analyzing the tree that the first level already separates the two classes fairly well. However, if a feature has a low feature_importance, it doesn’t mean that this feature is uninformative. It only means that the feature was not picked by the tree, likely because another feature encodes the same information. 

**Regression trees**
Decision tree for regression, is implemented using DecisionTreeRegressor and the usage and analysis of regression trees is similar to that of classification trees although the tree-based regression models are not able to extrapolate, or make predictions outside of the range of the training data.

**Example Consider**
the dataset of historical computer memory (RAM) prices over different years ranging from 1957 to 2015. The following figure shows the dataset, with the date on the x-axis and the price of one megabyte of RAM in that year on the y-axis: 

```
In[13]: import matplotlib.pyplot as plt 
import pandas as pd 
ram_prices = pd.read_csv("c:/Datasets/ram_price.csv")
plt.semilogy(ram_prices.date, ram_prices.price) 
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte") 
ram_prices
```

![image](https://user-images.githubusercontent.com/42868535/146066799-6e4377e2-a878-413d-9479-c4f871d40606.png)
![image](https://user-images.githubusercontent.com/42868535/146066867-c6908e9e-0ea4-4eaa-aa57-e6428f4402c6.png)


You can make a forecast for the years after 2000 using the historical data, with the date as the only feature. Make predictions on the whole dataset for visualization purpose only, but for a quantitative evaluation consider only the test dataset: 


```
In[14]: import numpy as np
from sklearn.tree import DecisionTreeRegressor 
# use historical data to forecast prices after the year 2000 data_train = ram_prices[ram_prices.date < 2000] 
data_test = ram_prices[ram_prices.date >= 2000]
# predict prices based on date X_train = data_train.date[:, np.newaxis] y_train = data_train.price tree = DecisionTreeRegressor().fit(X_train, y_train) # predict on all data X_all = ram_prices.date[:, np.newaxis] pred_tree = tree.predict(X_all) 
```
**The following diagram shows the predictions of the decision tree model:**

```
In[15]: 
plt.semilogy(data_train.date, data_train.price, label="Training data") 
plt.semilogy(data_test.date, data_test.price, label="Test data") 
plt.semilogy(ram_prices.date, pred_tree, label="Tree prediction") 
plt.legend() 
```
![image](https://user-images.githubusercontent.com/42868535/146067269-bc70b242-3739-4739-a538-9876d6c9becd.png)

The model makes perfect predictions on the training data i.e with no restriction on the complexity of the tree it learns the whole dataset. However, once you leave the data range for which the model has data, the model simply keeps predicting the last known point i.e. the tree has no ability to generate “new” responses, outside of what was seen in the training data.  

Notice that it is possible to make very good forecasts with tree-based models (for example, when trying to predict whether a price will go up or down). The point was to illustrate a particular property of how trees make predictions. 


**Strengths, weaknesses, and parameters**

The parameters that control model complexity in decision trees are the pre-pruning parameters that stop the building of the tree before it is fully developed. Usually, picking one of the pre-pruning strategies i.e. setting either max_depth, max_leaf_nodes, or min_samples_leaf are sufficient to prevent overfitting. Decision trees have two main advantages; the resulting model can easily be visualized and understood by non-experts (especially for smaller trees), and the algorithms are completely invariant to scaling of the data. As each feature is processed separately, and the possible splits of the data don’t depend on scaling, no preprocessing like normalization or standardization of features is needed for decision tree algorithms.  The main disadvantage of decision trees is that even with the use of pre-pruning, they tend to overfit and provide poor generalization performance.  







