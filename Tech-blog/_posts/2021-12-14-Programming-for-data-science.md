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










