---
layout: post
title:  "Machine Learning: Clustering Technique"
date:   2021-12-25 
category: tech
---

#### Introduction
`Clustering` or `cluster aalysis` is a machine learning technique, which groups the unlabelled dataset.

It can be defined as: ***A way of grouping the data ponts into different clusters, consisting of similar data points. The objects with possible similarities remain in a group that has less or no simolarities with another group***

It does this by finding some similar patterns in the unlabelled dataset such as shape or size, color, behaviour, etc and devides them as per the presence and absence of those similar patterns.

It is an `unsupervised learning` method, hence no supervision is provided to the algorithm, and it deals with the unlabeled datasets.

The clustering techniques is commonly used for **statistical data analysis**


#### The main goals of cluster analysis
- To get a meaningful intuition from the data we are working with.
- Cluster-then-predict where different models will be built for different subgroups.



#### Some of the common uses of clustering techniques:

1. Market segmentation.
2. Statistical data analysis
3. Social network analysis
4. Image segmentation
5. Anomaly detection. etc.

#### Specific uses of clustering technique:

1. It is used by the Amazon in its recommendation system to provide the recommendation system to provide the recommendations as per the past search of products.
2. Netflix also uses this technique to recommend the movies and web series to its users as per the watch history.

#### The diagram below illustrates the working of clustering algorithm




















#### Types of clustering Methods
The clustering methods are broadly divided into:
1. Hard clustering (datapoint belongs to only one group) 
2. Soft Clustering (data points can belong to another group also). 

Below are the main clustering methods used in Machine learning:
- Partitioning Clustering
- Density-Based Clustering
- Distribution Model-Based Clustering
- Hierarchical Clustering
- Fuzzy Clustering


**Partitioning clustering**
It is a type of clustering that devides the data into non-hierarchichal groups. It is also known as the `Centroid-based method`
In this type, the dataset is devided into a set of K groups, where K is used to define the number of pre-defined groups.

The cluster is centered in such a way that the distance between the data points of one cluster is minimum as compared to another cluster centroid. The most common example of partitioning clusterinng is the `K-Means clustering algorithm` 






















#### Density based clustering
The density based clustering method connects the highly-dense areas into clusters, and the arbitrary shaped distributions are formed as long as the dense regions can be connected.

This algorithm does this by identifying different clusters in the dataset and connects the areas of high densities into clusters.

The dense areas in data space are devided from each other by sparses areas.

These algorithms can face difficulty in clustering the data points if dataset has varying and high dimensions.
























#### Distribution Model-Based Clustering
In the distribution model-based clustering method, the data is divided based on the probability of how a dataset belongs to a particular distribution. The grouping is done by assuming some distributions commonly `Gaussian Distribution.`

The example of this type is the `Expectation-Maximization Clustering algorithm` that uses Gaussian Mixture Models (GMM).




















#### Hierarchical Clustering
Hierarchical clustering can be used as an alternative for the partitioned clustering as there is no requirement of pre-specifying the number of clusters to be created.

In this technique, the dataset is divided into clusters to create a tree-like structure, which is also called a `dendrogram`. The observations or any number of clusters can be selected by cutting the tree at the correct level.

The most common example of this method is the `Agglomerative Hierarchical algorithm`.

























#### Fuzzy Clustering
Fuzzy clustering is a type of soft method in which a data object may belong to more than one group or cluster. Each dataset has a set of membership coefficients, which depend on the degree of membership to be in a cluster.

`Fuzzy C-means algorithm` is the example of this type of clustering; it is sometimes also known as the Fuzzy k-means algorithm.



#### Popular Clustering algorithms that are widely used in machine learning

1. `K-Means algorithm:`
The k-means algorithm is one of the most popular clustering algorithms.
It classifies the dataset by dividing the samples into different clusters of equal variances. 
The number of clusters must be specified in this algorithm. It is fast with fewer computations required, with the linear complexity of O(n).

2. `Mean-shift algorithm:`
Mean-shift algorithm tries to find the dense areas in the smooth density of data points. 
It is an example of a centroid-based model, that works on updating the candidates for centroid to be the center of the points within a given region.

3. `DBSCAN Algorithm:`
It stands for Density-Based Spatial Clustering of Applications with Noise. 
It is an example of a density-based model similar to the mean-shift, but with some remarkable advantages.
In this algorithm, the areas of high density are separated by the areas of low density. 
Because of this, the clusters can be found in any arbitrary shape.

4. `Expectation-Maximization Clustering using GMM:`
This algorithm can be used as an alternative for the k-means algorithm or for those cases where K-means can be failed. 
In GMM, it is assumed that the data points are Gaussian distributed.

5. `Agglomerative Hierarchical algorithm:`
The Agglomerative hierarchical algorithm performs the bottom-up hierarchical clustering. 
In this, each data point is treated as a single cluster at the outset and then successively merged. 
The cluster hierarchy can be represented as a tree-structure.

6. `Affinity Propagation:`
It is different from other clustering algorithms as it does not require to specify the number of clusters. 
In this, each data point sends a message between the pair of data points until convergence. 
It has O(N2T) time complexity, which is the main drawback of this algorithm.





#### K-MEANS CLUSTERING
K-Means Clustering is an `Unsupervised Learning algorithm`, which groups the unlabeled dataset into different clusters. 

Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.

It is an `iterative algorithm` that divides the unlabeled dataset into k different clusters in such a way that each dataset belongs only one group that has similar properties.

It allows us to cluster the data into different groups and a convenient way to discover the categories of groups in the unlabeled dataset on its own without the need for any training.

It is a `centroid-based algorithm`, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters. The value of k should be predetermined in this algorithm.

The k-means clustering algorithm mainly performs two tasks:

1. Determines the best value for K center points or centroids by an iterative process.
2. Assigns each data point to its closest k-center. Those data points which are near to the particular k-center, create a cluster.

Hence each cluster has datapoints with some commonalities, and it is away from other clusters.


***How does the K-Means Algorithm Work?***
step01: Select the number K to decide the number of clusters.

step02: Select random K points or centroids.

step03: Assign each data point to their closest centroid, which will form the predefined K clusters.

step04: Calculate the variance and place a new centroid of each cluster.

step05: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

step06: If any reassignment occurs, then go to step-4 else go to FINISH.

step07: The model is ready.
 
#### Step-By-Step working of the K-means Clustering Algorithm.
Suppose we have two variables M1 and M2 as shown in the x-y axis scatter plot below,


























- Let's take number k of clusters, i.e., K=2, to identify the dataset and to put them into different clusters. 
It means here we will try to group these datasets into two different clusters.

- We need to choose some random k points or centroid to form the cluster. 
These points can be either the points from the dataset or any other point.
So, here we are selecting the two points shown below as k points, which are not the part of our dataset. 

























- Now we will assign each data point of the scatter plot to its closest K-point or centroid. 
- We will compute it by applying some mathematics that we have studied to calculate the distance between two points. 
So, we will draw a median between both the centroids as shown in the image below. 
























- From the above image, it is clear that points left side of the line is near to the K1 or blue centroid, and points to the right of the line are close to the yellow centroid. 
- Let's color them as blue and yellow for clear visualization.

























- As we need to find the closest cluster, so we will repeat the process by choosing a new centroid. 
- To choose the new centroids, we will compute the center of gravity of these centroids, and will find new centroids as shown below.



















- Next, we will reassign each datapoint to the new centroid.
- For this, we will repeat the same process of finding a median line. The median will be like the image below.



















image





















- From the image above, we can see, one yellow point is on the left side of the line, and two blue points are left to the line. 
- So, these three points will be assigned to new centroids.

- We will again repeat finding new centroids or K-points.
- so the new centroids will be as shown in the image below.
























- As we got the new centroids so again will draw the median line and reassign the data points. It will be as shown below



























image
























- We can see in the image above; there are no dissimilar data points on either side of the line, which means our model is formed. Consider the image below.



















- As our model is ready, so we can now remove the assumed centroids, and the two final clusters will be as shown in the image below;



















#### Choosing the value of `K` number of clusters using Elbow Method
The elbow method is used to determine the optimal number of clusters in k-means clustering. The elbow method plots the value of the cost function produced by different values of `k`. 

As you know, if `k` increases, average distortion will decrease, each cluster will have fewer constituent instances, and the instances will be closer to their respective centroids. 

However, the improvements in average distortion will decline as `k` increases. 
The value of `k` at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters.























To measure the distance between data points and centroid, we can use any method such as Euclidean distance or Manhattan distance.
To find the optimal value of clusters, the elbow method follows the below steps:

1. It executes the K-means clustering on a given dataset for different K values (ranges from 1-10).
2. For each value of K, calculates the `WCSS` value.
3. Plots a curve between calculated `WCSS` values and the number of clusters K.
4. The sharp point of bend or a point of the plot looks like an arm, then that point is considered as the best value of K.


#### ADVANTAGES OF K-MEANS
- It is very easy to understand and implement.
- If we have large number of variables then, K-means would be faster than Hierarchical clustering.
- On re-computation of centroids, an instance can change the cluster.
- Tighter clusters are formed with K-means as compared to Hierarchical clustering.


#### DISADVANTAGES OF K-MEANS
- It is a bit difficult to predict the number of clusters i.e. the value of k.
- Output is strongly impacted by initial inputs like number of clusters (value of k).
- Order of data will have strong impact on the final output.
- It is very sensitive to rescaling. If we will rescale our data by means of normalization or standardization, then the output will completely change.final output.
- It is not good in doing clustering job if the clusters have a complicated geometric shape.


#### APPLICATIONS OF K-MEANS CLUSTERING ALGORITHM
1. In Identification of Cancer Cells:
The clustering algorithms are widely used for the identification of cancerous cells. 
It divides the cancerous and non-cancerous data sets into different groups.

2. In Search Engines:
Search engines also work on the clustering technique. 
The search result appears based on the closest object to the search query. 
It does it by grouping similar data objects in one group that is far from the other dissimilar objects. 
The accurate result of a query depends on the quality of the clustering algorithm used.

3. Customer Segmentation:
It is used in market research to segment the customers based on their choice and preferences.

4. In Biology:
It is used in the biology stream to classify different species of plants and animals using the image recognition technique.

5. In Land Use:
The clustering technique is used in identifying the area of similar lands use in the GIS database. 
This can be very useful to find that for what purpose the particular land should be used, that means for which purpose it is more suitable.



***While working with K-means algorithm we need to take care of the following things***
- While working with clustering algorithms including K-Means, it is recommended to standardize the data because such algorithms use distance-based measurement to determine the similarity between data points.
- Due to the iterative nature of K-Means and random initialization of centroids, K-Means may stick in a local optimum and may not converge to global optimum. That is why it is recommended to use different initializations of centroids.
