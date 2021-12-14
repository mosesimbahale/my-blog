---
layout: post
title:  "Programming for data science"
date:   2021-12-14 10:30:35 
category: tech
---

### Introduction: 

The programming requirements of data science demands a very versatile yet flexible language which is simple to write the code but can handle 
highly complex mathematical processing. Python is most suited for such requirements as it has already established itself both as a language for general 
computing as well as scientific computing. Moreover it is being continuously upgraded in form of new addition to its plethora of libraries aimed at different programming requirements.

### Features of python which makes it the preferred language for data science:

- A simple and easy to learn language which achieves result in fewer lines of code than other similar languages like R.
- It is cross platform, so the same code works in multiple environments without needing any change.
- It executes faster than other similar languages used for data analysis like R and MATLAB. 
- Its excellent memory management capability, especially garbage collection makes it versatile in gracefully managing very large volume of data transformation, slicing, dicing and visualization. 
- Python has got a very large collection of libraries which serve as special purpose analysis tools. E.g. NumPy package deals with scientific computing and its array needs much less memory than the conventional python list for managing numeric data. Also the number of such packages is continuously growing. 
- Python has packages which can directly use the code from other languages like Java or C. This helps in optimizing the code performance by using existing code of other languages, whenever it gives a better result. 


### Python Machine Learning Ecosystem 

The Python machine learning ecosystem is a collection of libraries that enable the developers to extract and transform data, perform data wrangling operations, apply existing robust Machine Learning algorithms and also develop custom algorithms easily. These libraries include numpy, scipy, pandas, scikit-learn, statsmodels, tensorflow, keras, etc. The following is a description of these libraries: 

1. PANDAS:- used for Data Analysis 
2. NUMPY: - used for numerical analysis and formation i.e. for matrix and vector manipulation 
3. MATPLOTLIB: - used for data visualization  
4. SCIPY: - used for scientific computing 
5. SEABORN: - used for data visualization 
6. TENSORFLOW: - used  in deep learning 
7. SCIKIT-LEARN: - used in machine learning i.e. used as a source for many machine learning algorithms and utilities 
8. KERAS : - used for neural networks and deep learning 

  

### Components of Python Machine Learning Ecosystem
1. Jupyter Notebook - The Jupyter Notebook, formerly known as ipython notebooks is an interactive environment for running code in the browser. It is a great tool for exploratory data analysis and is widely used by data scientists.    

2. NumPy - Numpy is the backbone of Machine Learning in Python. It is one of the most important libraries in Python for numerical computations. It’s used in almost all Machine Learning and scientific computing libraries. It stands for Numerical Python and provides an efficient way to store and manipulate multidimensional arrays in Python. Numerical Python and provides an efficient way to store and manipulate multidimensional arrays in Python. Generally, NumPy can also be seen as the replacement of MatLab because NumPy is mostly used along with Scipy (Scientific Python) and Mat-plotlib (plotting library).

***Numpy ndarray*** 
The numeric functionality of numpy is orchestrated by two important constituents of the numpy package, ndarray and Ufuncs (Universal function).   ndarray (simply arrays or matrices) is a multi-dimensional array object which is the core data container for all of the numpy operations. Mostly an array will be of a single data type (homogeneous) and possibly multi-dimensional.   Universal functions are the functions which operate on ndarrays in an element by element fashion.

EXAMPLE:
`creating an array`
Arrays can be created in multiple ways in numpy. A single dimensional array can be created from Python lists using np.array() method as shown below:

> In[3]: arr = np.array([1,3,4,5,6]) 

The shape attribute of the array object returns the size of each dimension in the form of (row, columns), while the size returns the total size of the array: 

>  In [4]: arr.shape 
>
>  Out[4]: (5,) 

Unlike Python lists, NumPy arrays can explicitly be multidimensional. A multidimensional array is created as shown below: 

> ' In[5]: import numpy as np
> '        x = np.array([[1, 2, 3], [4, 5, 6]]) 
> '        print("x:\n{}".format(x)) 

> ' Out[5]: 
> '    x: [[1 2 3] 
> '   [4 5 6]] 

***creating arrays from scratch***
For large arrays it is more efficient to create arrays from scratch using a bunch of special functions built in NumPy as shown in the following examples: 
