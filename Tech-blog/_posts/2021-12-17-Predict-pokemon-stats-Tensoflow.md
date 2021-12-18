---
layout: post
title:  "Tensorflow: Predict pokemon stats"
date:   2021-12-17 
category: tech
---


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
Reasons why we need a recap

- Some categories are not numerical.
- Converting multiple categories into numbers implies that they are on a scale.
- The categories for "Type_1" should all be boolean (0 or 1).


