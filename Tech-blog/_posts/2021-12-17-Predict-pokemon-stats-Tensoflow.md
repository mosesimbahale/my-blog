---
layout: post
title:  "Tensorflow: Predict pokemon stats"
date:   2021-12-17 
category: tech
---

![image](https://user-images.githubusercontent.com/42868535/146635309-7a94db40-7136-41c2-86da-fc8e30da9f2f.png)


Welcome to the world of machine learning with TensorFlow! Working with TensorFlow can seem intimidating at first, but this tutorial will start with the basics to ensure you have a strong foundation with the package. This tutorial will be focusing on classifying and predicting Pokémon, but the elements discussed within it can certainly be helpful when using TensorFlow for other ideas, as well. Without further ado, let's begin!


First, let's download TensorFlow through `pip.` While you can install the version of TensorFlow that uses your GPU, we'll be using the CPU-driven TensorFlow. Type this into your terminal:

`pip install tensorflow`

Now that it's installed, we can truly begin. Let's import Tensorflow, and a few other packages we'll need. All of this course involve using the command line interface. Enter these commands to import and the necessary packages:

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
```

The dataset we'll be using is the compilation of stats and traits for the Pokémon video games. Pokémon is a popular game for generations of Nintendo handheld video game consoles where players collect and train animal-like creatures called Pokémon. We'll be creating a model to try to predict whether a Pokémon is a legendary Pokémon, a rare type of Pokémon who's the only one of its species.


There are a lot of existing compilations of Pokémon stats, but we'll be using a .CSV version [found on Kaggle.](https://www.kaggle.com/alopez247/pokemon) There's a download button on the website, so save the file to your computer and we can begin.

First, we need to read in the CSV file. We'll be doing so using Pandas:

`df = pd.read_csv('/path/to/file/pokemon.csv')`

First, let's see what the categories of data are. This was also available on the Kaggle page, but that won't be the case for most real-world data:


```
df.columns
>>> Index(['Number', 'Name', 'Type_1', 'Type_2', 'Total', 'HP', 'Attack',
       'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Generation', 'isLegendary',
       'Color', 'hasGender', 'Pr_Male', 'Egg_Group_1', 'Egg_Group_2',
       'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Catch_Rate',
       'Body_Style'],
      dtype='object')

```

Okay so we have a lot of types of data here! Some of these descriptions might be confusing to those who aren't very familiar with the games. That's okay, we'll narrow our focus a little and only select categories we think will be relevant. It's always nice to have more data to train the model with, but it also takes time to clean and prepare that data. We'll be keeping it simple here:

`df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]`

> **Note We used "pandas". 🐼 to read our csv file.**

Now that we can see all of our data, we'll need to format it to be read by the model. First, we need to make sure all the data is numerical. A lot of our data already is, such as stats like 'HP' and 'Attack'. Great!

A few of the categories aren't numerical however. One example is the category that we'll be training our model to detect: the "isLegendary" column of data. These are the labels that we will eventually separate from the rest of the data and use as an answer key for the model's training. We'll convert this column from boolean "False" and "True" statements to the equivalent "0" and "1" integers:

`df['isLegendary'] = df['isLegendary'].astype(int)`

There are a few other categories that we'll need to convert as well. Let's look at "Type_1" as an example. Pokémon have associated elements, such as water and fire. Our first intuition at converting these to numbers could be to just assign a number to each category, such as: Water = 1, Fire = 2, Grass = 3 and so on. This isn't a good idea because these numerical assignments aren't ordinal; they don't lie on a scale. By doing this, we would be implying that Water is closer to Fire than it is Grass, which doesn't really make sense.

The solution to this is to create dummy variables. By doing this we'll be creating a new column for each possible variable. There will be a column called "Water" that would be a 1 if it was a water Pokémon, and a 0 if it wasn't. Then there will be another column called "Fire" that would be a 1 if it was a fire Pokémon, and so forth for the rest of the types. This prevents us from implying any pattern or direction among the types. Let's do that:

```
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)
```

This function first uses `pd.get_dummies` to create a dummy DataFrame of that category. As it's a seperate DataFrame, we'll need to `concat`enate it to our original DataFrame. And since we now have the variables represented properly as separate columns, we `drop` the original column. Having this in a function is nice because we can quickly do this for many categories:

`df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])`



# **Recap**
Reasons why we need to create dummy variables:

- Some categories are not numerical.
- Converting multiple categories into numbers implies that they are on a scale.
- The categories for "Type_1" should all be boolean (0 or 1).



# **Split and Normalize Data**

Now that we have our data in a useable form, we need to split it. We want to have a set of data that we'll use to train our model, and we'll use another set of data to test our model after we've trained it. In general, the data is randomly split with about 70% being used for training and 30% used for testing. For easier visualization, we'll be splitting the data by Pokémon generation. The first generation of Pokémon (from Pokémon Red, Blue, and Yellow) will be our testing data while the rest will be our training data:


```
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)

df_train, df_test = train_test_splitter(df, 'Generation')
```

This function takes any Pokémon whose "Generation" label is equal to 1 and putting it into the test dataset, and putting everyone else in the training dataset. It then `drops` the `Generation` category from the dataset.


Now that we have our two sets of data, we'll need to separate the labels (the 'islegendary' category) from the rest of the data. Remember, this is the answer key to the test the algorithms are trying to solve, and it does no good to have them learn with the answer-key in (metaphorical) hand:


```
def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)
```

This function extracts the data from the DataFrame and puts it into arrays that TensorFlow can understand with`.values.` We then have the four groups of data:


`train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')`


Now that we have our labels extracted from the data, let's normalize the data so everything is on the same scale:

```
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normalizer(train_data, test_data)
```

Now we can get to the machine learning! Let's create the model using Keras. Keras is an API for Tensorflow. We have a few options for doing this, but we'll keep it simple for now. A model is built upon layers. We'll add two fully connected neural layers.

The number associated with the layer is the number of neurons in it. The first layer we'll use is a 'ReLU' (Rectified Linear Unit)' activation function. Since this is also the first layer, we need to specify `input_size`, which is the shape of an entry in our dataset.

After that, we'll finish with a softmax layer. Softmax is a type of logistic regression done for situations with multiple cases, like our 2 possible groups: 'Legendary' and 'Not Legendary'. With this we delineate the possible identities of the Pokémon into 2 probability groups corresponding to the possible labels:


```
length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))
```


# **Compile and Evaluate Model**

Once we have decided on the specifics of our model, we need to do two processes: Compile the model and fit the data to the model.

We can compile the model like so:
`model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`

Here we're just feeding three parameters to `model.compile.` We pick an optimizer, which determines how the model is updated as it gains information, a loss function, which measures how accurate the model is as it trains, and metrics, which specifies which information it provides so we can analyze the model.

The optimizer we're using is the Stochastic Gradient Descent (SGD) optimization algorithm, but there are others available. For our loss we're using sparse_categorical_crossentropy. If our values were one-hot encoded, we would want to use "categorial_crossentropy" instead.

Then we have the model fit our training data:


`model.fit(train_data, train_labels, epochs=400)`

The three parameters `model.fit` needs are our training data, our training labels, and the number of epochs. One epoch is when the model has iterated over every sample once. Essentially the number of epochs is equal to the number of times we want to cycle through the data. We'll start with just 1 epoch, and then show that increasing the epoch improves the results.

# **Test**
Now that the model is trained to our training data, we can test it against our training data:

```
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value})'
>>> Our test accuracy was 0.980132
```




`model.evaluate` will evaluate how strong our model is with the test data, and report that in the form of loss value and accuracy value (since we specified `accuracy` in our `selected_metrics` variable when we compiled the model). We'll just focus on our accuracy for now. With an accuracy of ~98%, it's not perfect, but it's very accurate.

We can also use our model to predict specific Pokémon, or at least have it tell us which status the Pokémon is most likely to have, with `model.predict.` All it needs to predict a Pokémon is the data for that Pokémon itself. We're providing that by selecting a certain `index` of `test_data`:

```
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{test_labels[index]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return(prediction)
```


Let's look at one of the more well-known legendary Pokémon: Mewtwo. He's number 150 in the list of Pokémon, so we'll look at index 149:

```
predictor(test_data, test_labels, 149)
>>> This was correctly predicted to be a "1"!

```

Nice! It accurately predicted Mewtwo was a legendary Pokémon.
