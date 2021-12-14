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

![image of creating array](https://user-images.githubusercontent.com/42868535/146039626-fa02df64-f58a-4929-8c9a-e5b00eacba4f.png)

***np.ones:*** Creates a matrix of specified dimension containing only ones:

![image of creating np.ones](https://user-images.githubusercontent.com/42868535/146039827-35e7c99b-a309-4c29-b835-f759609e96f5.png)

***np.arange:*** creates an array filled with a linear sequence, starting at 0, ending at 20, stepping by 2. This is similar to the built-in range() function  

![image ofnp.range](https://user-images.githubusercontent.com/42868535/146040226-1d20d13c-2599-49e7-9c3a-05af1a0a2a02.png)

***np.identity:*** Creates an identity matrix of specified dimensions: 

![image of np.identity](https://user-images.githubusercontent.com/42868535/146040415-33f20cb2-e73c-4c8d-8a00-7809293dd09c.png)

To initialize an array of a specified dimension with random values can be done by using the randn function from the numpy.random package: 

![image of random values](https://user-images.githubusercontent.com/42868535/146040728-414ab46c-dae1-4afa-bc97-e8b050c5f329.png)



3. Pandas is a Python library for data wrangling and analysis. It is built around a data structure called the DataFrame that is modeled after the R DataFrame i.e. it similar to an Excel spreadsheet. Pandas provides a great range of methods to modify and operate on a table; in particular, it allows SQL-like queries and joins of tables. In contrast to NumPy, which requires that all entries in an array be of the same type, pandas allows each column to have a separate type. Another valuable tool provided by pandas is its ability to ingest from a great variety of file formats and databases, like SQL, Excel files, and comma-separated values (CSV) files.  The following is an example of creating a DataFrame using a dictionary: 

![image](https://user-images.githubusercontent.com/42868535/146041110-119d1d3c-5ade-4192-a8bb-b274877e201b.png)


***Data Processing in Pandas***

Using Pandas, you can process data using the following five steps:  
i) Load 
ii) Prepare  
iii) Manipulate  
iv) Model  
v) Analyze  


**Data Structures of Pandas**
All the data representation in pandas is done using two primary data structures:

i) **Series** 
Series in pandas is a one-dimensional ndarray with an axis label. i.e. its functionality is similar to a simple array. The values in a series will have an index that needs to be hashable. This requirement is needed when we perform manipulation and summarization on data contained in a series data structure.  

ii) **Dataframe**
 Dataframe is the most important and useful data structure, which is used for almost all kind of data representation and manipulation in pandas. Pandas are extremely
 useful in representing raw datasets as well as processed feature sets in Machine Learning and Data Science. All the operations can be performed along the axes, rows, and columns, in a dataframe. 

 **Data Retrieval**
Pandas provides numerous ways to retrieve and read in data. You can convert data from CSV files, databases, flat files, etc into dataframes. You can also convert a list of dictionaries (Python dict) into a dataframe. The following are the most important data sources: 
  
i) **List of Dictionaries to Dataframe:** This is one of the simplest methods to create a dataframe. It is useful in scenarios when you arrive at the data you want to analyze, after performing some computations and manipulations on the raw data. It allows integration of a panda based analysis into data being generated by other Python processing pipelines. 

ii) **CSV Files to Dataframe** This is perhaps one of the most widely used ways of creating a dataframe. You can easily read a CSV, or any delimited file (like TSV), and convert it into a dataframe using pandas. The following is a sample slice of a CSV file containing the data of cities of the world from [here](http://simplemaps.com/data/world-cities.) 


![image](https://user-images.githubusercontent.com/42868535/146045199-a0a60128-037e-45fd-9c7b-8987e8a71fb2.png)

The data is obtained using the following code 

![image](https://user-images.githubusercontent.com/42868535/146045308-a1aa2dc4-d0ea-4bbd-9142-6ea23a532220.png)

**Databases to Dataframe**

The most important data source for data scientists is the existing data sources used by their organizations. Relational databases (DBs) and data warehouses are the de facto standard of data storage 
Data Science Programming~ Wainaina                                                                                                               Page 9 of 11 in almost all organizations. Pandas provides capabilities to connect to these databases directly, execute queries on them to extract data, and then convert the result of the query into a structured dataframe. The pandas.from_sql function combined with Python’s powerful database library implies that the task of getting data from DBs is simple and easy.  

**Example**
The following code is used read data from a Microsoft SQL Server database.

```
server = 'xxxxxxxx' # Address of the database server
user = 'xxxxxx' # the username for the database server 
password = 'xxxxx' # Password for the user
database = 'xxxxx' # Database in which the table is present
conn = pymssql.connect(server=server, user=user, password=password, 
database=database)
query = "select * from some_table" 
df = pd.read_sql(query, conn) 
```
conn is an object used to identify the database server information and the type of database to pandas 

4. Matplotlib 
matplotlib is the primary scientific plotting library in Python. It provides functions for making publication-quality visualizations such as line charts, histograms, scatter plots, etc. Visualizing your data and different aspects of your analysis can give you important insights.   
When working inside the Jupyter Notebook, you can show figures directly in the browser by using the %matplotlib notebook and %matplotlib inline commands. 

**Example:**
The following code produces the plot

```
In [1]: 
%matplotlib inline
import matplotlib.pyplot as plt 
#or you can use "from matplotlib import pyplot as plt"

# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine 
y = np.sin(x) # The plot function makes a line chart of one array against another 
plt.plot(x, y, marker="x") 
```

![image](https://user-images.githubusercontent.com/42868535/146046857-554371c5-fb92-45e2-9e87-9fe772fb2234.png)



5. SciPy 

SciPy is a collection of functions for scientific computing in Python. It provides, among other functionality, advanced linear algebra routines, mathematical function optimization, signal processing, special mathematical functions, and statistical distributions. scikit-learn draws from SciPy’s collection of functions for implementing its algorithms. One of the most important part of SciPy is scipy.sparse, which provides sparse matrices, which are another representation that is used for data in scikitlearn. Sparse matrices are used whenever we want to store a 2D array that contains mostly zeros. 

```
In[1]: 
from scipy import sparse 
# A 2D NumPy array with a diagonal of ones, and zeros everywhere else 
eye = np.eye(4) 
print("NumPy array:\n{}".format(eye)) 
```


It produces the following output
```
Out[2]:
SciPy sparse CSR matrix:
(0, 0) 1.0
(1, 1) 1.0
(2, 2) 1.0
(3, 3) 1.0 
```


6. Scikit-learn 
Scikit-learn is one of the most important and indispensable Python frameworks for Data Science and Machine Learning in Python. It is built on top of the NumPy and SciPy scientific Python libraries and implements a wide range of Machine Learning algorithms covering major areas of Machine Learning like classification, clustering, regression, etc. All the mainstream Machine Learning algorithms like support vector machines, logistic regression, random forests, K-means clustering, hierarchical clustering, etcs are implemented efficiently in this library. Perhaps this library forms the foundation of applied and practical Machine Learning. Besides this, its easy-to-use API and code design patterns have been widely adopted across other frameworks. 


# DATA REPRESENTATION IN SCIKIT LEARN

Machine learning is about creating models from data. It is important then to understand how data can be represented in order to be understood by the computer. The best way to think about data within Scikit-Learn is in terms of tables of data. 


**Data as table** 
A basic table is a two-dimensional grid of data, in which the rows represent individual elements of the dataset, and the columns represent quantities related to each of these elements. 

#### Features matrix 

The table layout makes the information to be represented as a two dimensional numerical array or matrix, which is called the features matrix. By convention, this features matrix is often stored in a variable named X. Generally, features matrix is assumed to be two-dimensional, with shape [n_samples, n_features], and is most often contained in a NumPy array or a Pandas DataFrame, though some Scikit-Learn models also accept SciPy sparse matrices.  
The samples (rows) always refer to the individual objects described by the dataset. The sample might be, a person, a document, an image, a sound file, a video, or anything else that can be described with a set of quantitative measurements. 
The features (columns) refer to the distinct observations that describe each sample in a quantitative manner. Features are generally real-valued, but may be Boolean or discrete-valued in some cases. 

#### Target array 

In addition to the feature matrix X, you also need to work with a label or target array that by convention is called y. This target array is usually one dimensional, with length n_samples, and can be contained in a NumPy array or Pandas Series. It may have continuous numerical values, or discrete classes/labels. While some Scikit-Learn estimators do handle multiple target values in the form of a two-dimensional [n_samples, n_targets] target array, you will be working with the a one-dimensional target array. 

Notice that the distinguishing feature of the target array is the quantity that needs to be predicted from the data (i.e dependent variable). 

#### Scikit-Learn’s Estimator API 

Every machine learning algorithm in Scikit-Learn is implemented via the Estimator API, which provides a consistent interface for a wide range of machine learning applications. 

#### Basics of the API
The following are the steps commonly used in the Scikit-Learn estimator API: 
i). Choose a class of model by importing the appropriate estimator class from Scikit-Learn. 
ii). Choose model hyperparameters by instantiating the class with desired values. 
iii). Arrange data into a features matrix and target vector 
iv). Fit the model to your data by calling the fit() method of the model instance. 
v). Apply the model to new data:  

- For supervised learning, you predict labels for unknown data using the predict() method. 
- For unsupervised learning, you transform or infer properties of the data using the transform() or predict() method.

vi). Check the results of model fitting to know whether the model is satisfactory 


# Simple linear regression 

Linear regression involves coming up with a straight-line fit to data. A straight-line fit is a model of the form     y = ax + b where a is commonly known as the slope, and b is commonly known as the intercept. 

**Example**
Consider a simple linear regression that involves fitting a line to x, y data. Use the following data sample:

```
In[1]: import matplotlib.pyplot as plt
import numpy as np 
rng = np.random.RandomState(42) 
x = 10 * rng.rand(40) 
y = 2 * x - 1 + rng.randn(40)
print (x) 
print (y) 
plt.scatter(x, y);
```

![image](https://user-images.githubusercontent.com/42868535/146052664-4d14d97f-066c-436f-8bd6-4eb29f1fec58.png)


With this data, you can use the steps outlined for the Scikit-Learn estimator API above: 


1. **Choose a class of model.**  In Scikit-Learn, every class of model is represented by a Python class. E.g., for a simple linear regression model, you import the linear regression class as shown below: `In[2]: from sklearn.linear_model import LinearRegression` 

2. **Choose model hyper parameters.** Depending on the model class you are working with, you might need to consider one or more of the following options:
- Would you like to fit for the offset (i.e., intercept)?  
- Use fit_intercept  (True by default) that decides whether to calculate the intercept 푏₀ (True) or consider it equal to zero (False).
- Would you like the model to be normalized? Use normalize (False by default) that decides whether to normalize the input variables (True) or not (False).  
- Would we like to pre-process the features to add model flexibility? 
- What degree of regularization would we like to use in the model?  
- How many model components would we like to use?
 
These are important choices that must be made once the model class is selected. These choices are often represented as hyperparameters, or parameters that must be set before the model is fit to data. This is done when you choose hyperparameters by passing values at model instantiation. For  instance, you can specify you would like to fit the intercept using the fit_intercept hyperparameter: 

``` 
In[3]: 
model = LinearRegression(fit_intercept=True)
Model
Out[3]: LinearRegression() 
```
>Notice that when the model is instantiated, the only action is the storing of the hyperparameter values. In particular, the model has not yet been applied to any data 

3. **Arrange data into a features matrix and target vector.** 
Scikit-Learn data representation, requires a two-dimensional features matrix and a one-dimensional target array. The target variable y is in the correct form (length-n_sample array), but you need to reshape data x into one-dimensional array i.e. make it a matrix of size [n_samples, n_features]. 

``` 
In[4]:X = x[:, np.newaxis] 
X.shape 
Out[4]: (40, 1) 
```

4. **Fit the model to your data.** 
Apply the model to data. This can be done with the fit() method of the model as follows: 

```
In[5]: model.fit(X, y) 
Out[5]: 
LinearRegression()
```
The fit() command causes a number of model-dependent internal computations to take place, and the results of these computations are stored in model specific attributes. In this linear model, we have the following: 

```
In[6]: model.coef_ 
Out[6]: array([2.0133901]) 
In[7]: model.intercept_ 
Out[7]: -1.139509001948376
```
These two parameters represent the slope and intercept of the simple linear fit to the data.


5. **Predict labels for unknown data.**
Once the model is trained, the main task of supervised machine learning is to evaluate it based on what it says about new data that was not part of the training set. In Scikit-Learn, this can done by using the predict() method. In this example, the “new data” will be a grid of x values, and we will need to know what y values the model predicts: 
`In[8]: xfit = np.linspace(-1, 11, 40) `

Change the x values into a [n_samples, n_features] features matrix format and which can be feed to the model as shown below: 
```
In[9]: Xfit = xfit[:, np.newaxis] 
yfit = model.predict(Xfit)
```
Finally, visualize the results by plotting first the raw data, and then the model fit: 

```
In[10]: plt.scatter(x, y) 
plt.plot(xfit, yfit, color='red') 
plt.xlabel("xfit") 
plt.ylabel("yfit"); 
```

![image](https://user-images.githubusercontent.com/42868535/146055435-0228643f-3dc4-4a95-9e98-024cda944566.png)




6. Check the results of model fitting to know whether the model is satisfactory 
