---
layout: post
title:  "ML: Data Exploration, Visualization & Pre-processing"
date:   2021-12-25 
category: tech
---

## Outline
## Data exploration
- Types of data
- Descriptive statistics
- Visualization

## Data preprocessing
- Aggregation
- Sampling
- Dimensionality reduction
- Feature creation
- Discretization
- Attribute transformation
- Compression
- Concept hierarchies





## Getting started

## Types of data sets

1. Record
- Data matrix
- Documennt data
- Transaction data

**Record data:** Cosists of a collection of records, each of which consists of a fixed set of attributes

**Data matrix:** If data objects have the same fixed set of numeric attributes, then the data objeccts can be thought of as points in a multi-dimensional space, where each dimension represents a distinct attribute.

Such data sets can be represented by an m by n matrix, where there are m rows, one for each object, and n columns, one for each attribute.

**Document data:** Each document beomes a `term` vector,

 - Each term is a component(attribute) of the vector,
 - The value of each componet is the number of times the corresponding term occurs in the document

 **Transaction data:** A special type of data where;
  - Each record(transaction) ivolves a set of items.
  - For example, consider a grocery store. The set of products purchased by a customer during one shopping trip constitute a transaction, while the individual products that were purchased are the items.





2. Graph
- World wide web
- Molecular structure




3. Ordered
- Spatial data
- Temporal data
- Sequential data
- Genetic sequence data









## Important characteristics of strictured data

1. Dimensionality
- Curse of dimensionality
2. Sparity
- Only presence counts
3. Resolution
- Patterns depends on scale



### What is data exploration?

A pleriminary exploration of data to better understand its characteristics.

Key motivations of dat exploration include
- Helping to select the right tol for preprocessing or analysis.
- Making use of humans' abilities to recognize patterns.
- People can recognize patterns not captured by dat analysis tools.

Ralated to the area of Exporatory Data Analysis (EDA).
- Created by statistical John Tukey.
- Seminal book is eploratory data analysis by Tukey.
- A nice online introduction can be found in chapter one of the NIST Engineering Statist Handbook, can be founf [here](http://www.iti.nist.gov/div898/handbook/index.htm)



## Techniques used in data exploration

The EDA, as originally defined by Tukey;
- Focuses on visualization, Clustering and anomaly detection were viewed as exploratory techniques.

- In data mining, clustering and anomaly detection are major areas of intres, and not thought of just as exploratory.

- In this discussion of data exploration we focus on `summary statistics`, `visualization` & `Online Analytical Processing`(OLAP).

**Iris Smple data set:**
- Many of the exploratory data techniques are illustrated with the [Iris plant data set.](htt://www.ics.uci.edu/~mlearn/MLRepository.html)

From the statistician Douglas Fisher,
- Three flower types `classes`: Setosa, Virginica & Versicolour 
- Four `non-class` attributes: Sepal, width & length.


## Mining data descriptive characteristics

Motivaion
- To better understand the data: Central tendency, variation & spread.
- Data dispersion characteristics: Median, Max, Min, Quantiles, Outliers, Variance, etc.

Numerical dimensions corresponding to sorted intevals
- Data dispersion: analyzed with multiple granularities of precision.
- Boxplot or quantile analysis on the sorted intervals.

Disperison analysis on computed measures.
- Folding measures into numerical dimensions.
- Boxplot or quantile analysison the transformed cube.

## Summary Statistics
- They are numbers that summarize properties of the data.
- Summarized properties include frequency, location & spread.

Examples:
- Location- mean
- spread- standard deviation

Most summary statistics can be calculated ina single pass through the data


**Frequency and Mode**
The frequency and mode of an attribute value is the percenage of time the value occurs in the data set. For example, given the attribute `gender` and a representative population of the people, the gender `female` occurs about 50% of the time.

The mode of an attribute is the most fequent attribute value.

The notion of frequency and mode are typically used with categorical data.


**Percentiles**




