---
layout: post
title:  "Machine Learning: Training Models"
date: 2021-12-25 
category: tech
---

#### Linear Regression Model

##### Two different ways of training linear regression model
1. Using a `direct "close-form" equation` that directly computes the model parameters that best fit the model to the training set(i.e., the model parameters that minimize the cost function over the training set).

2. Using an `iterative optimization approach` called Gradient Descent(GD), that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as this first method.


##### Using a direct closed-form equation
- Here we directly computes the model parameters that best fit the model to the training set(i.e., the model parameters that minimize the cost function over the training set).

- So far we have talked about simple linear regression model such as:













##### So how deo we train it then?
Understand that training a model means setting its parameters so that the model best fits the training set. Therefore we first need to measure of how well(or poorly) the model fits the training data.

Do you remember the performance evaluation measures metrics for regression such as RMSE, MAE etc ?

Therefore, to train a linear regression model, you need to find the value of `theta` that minimizes the RMSE

In practice, it is simpler to minimize the Mean Square Error(MSE) than the (RMSE) and it leads to the same results (because the value that minimizes a function allso minimizes its square root).

The MSE of a linear regression hypothesis ha on a training ser `X is calculated using the below equation:`










This is the MSE cost function for s linear regression model.

- The only difference is that we write h0 instead of just h in order to make it clear that the model is parametirized by the vector 0.

- To simplify notations, we will just write (`MSE(0)`) Instead of `MSE(X,h0)`.


##### The normal equation
To find the value of 0 that minimizes the cost function, there is a `closed-form solution`- a mathematical equationthat gives the result directly. This is called the `normal equation`. It is as shown below:









0 is the value that minimizes the cost function.
y is the vector of target values containing `y(1)` to `y(m)`.

#### Example
- Lets randomly generate some linear-looking data to test this equation:


































































































#### Computational complexity
The normal equation computes the inverse of X











The computational complexity of inverting such as a matrix is typically about 





If you double the number of features, you multiply the computation time by roughly 



On the positive side, this equation is linear with regards to the number of instances in the training ser=t therefore it handles large training sets efficiently, provided they can fit in memory. However, as the number of features grows large( e.g 100,000), the normal the normal uquations gets very slow.

Once you have trained your linear regression model (using the normal equation or any other algorithm), predictions are very fast: the computational complexity is linear with regards to both the number of instances you want to make predictions on and the number of features.

In other words, making predictions on twice as many instances(or twice as many features) will just take roughly twice as much time.

#### Gradient descent
Gradient descent is a very generic optimized algorithm capable of finding optimal solutions to a wide range of problems.

Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize machine learning models.

These algorithms, however, are often used as black-box optimizers, as practical explnations of their strengths and weaknesses are hard to come by.

The general idea of gradient descent is to tweak parameters iteratively in order to minimize a cost function.

Gradient descent is used to models where there are a large number of features, or too many trainig instances to fit in memory.


#### Scenario on how gradient descent works
Suppose you are in the mountains in a dense fog; you can only feel the slope of the ground below your feet. A good strategy to get to the bottom of the valley quickly is to go downhill in the direction of the steepest slope.

This is exactly what gradient descent does: It measures the local gradient of the error function with regards to the parameter vector `0`, and it goes in the direction of descending gradient.

Once the gradient is zero, you have reached a minimum gradient.


#### Gradient descent
Concretely, you start by filling 0 with random values(this is called random initializations), and then you improve it gradually, taking one baby step ata time, each step attempting to decrease the cost function (e.g., the MSE), untik the algorithm converges to a minimum


















#### Learning rate
An important parameter in gradient descent is the size of the steps, determined by the learning rate hyperparameters.

If the learning rate is too small, then the algorithm will have to go through many iterations to coonverge, which will take a long time.

On the other hand, if the learning rate is too high, you might jump across the valley and end up on the other side, possibly even higher up than you were before. 

This might make the algorithm diverge, with larger and larger values, failing to find a good solution.
























#### Issues with learning rates in gradient descent
However, not all cost functions look like nice regular bowls. There may be holes, ridges, plateaus, and all sorts of irregular terrains, making convergence to the minimum very difficult as shown below: 




















If the random initialization starts the algorithm on the left, then it will converge to a local minimum, which is not as good as the global gllobal minimum. It it starts on the right, then it will take very long time to cross the plateau, and if you stop too early you will never reach the global minimum.

When using gradient descent, you should ensure that all features have a similar(e.g., using scikit-learn's standardScalar Class), or else it will take much longer to converge.

The more parameters a model has, the more dimensions this space has, and the harder the search is: Searching for a needle in a 300-dimensional haystack is much trickier than in three dimension.

However, in the case of linear regression, the cost function is conves, hence , the needlle is simply at the bottom of the bowl.





#### Batch gradient descent
This is a type of gradinet descent which processes all the training examples for each iteration of gradient descent. To implement gradient descent, you need to compute the gradient of the cost funcion with regards to each model parameter `0` i.e., how much the cost function will change if you change `0` just a little bit. This is called a partial derivative.

A good example to explain partial derivatives is like asking "what is the slope of the mountain under my feet if i face east ?" and then asking the same question facing north(and so on for all other dimensions, if you can imagine a unive)























- The above equation computes the partial derivative of the cost function with regards to parameter 0 denoted as


















Instead of computing these gradients individually, you can use equation below to compute them all in one go.













The gradient vector, noted `MSE(0), contains all the partial derivatives of the cost function (One for each model parameter)

Notice that this formula involves calculations over the full training set X, `at each gradient descent step! This is why the algorithm is called` `batch gradient descent`
  

It uses the whole batchof training data at every step. As a result is it terribly slow on very large traaining sets (But we will see much faster gradient descent algorithms shortly).

However, gradient descent scales well with the number of features; training a linear regression model when there are hundreads of thousands of features is much faster using gradient descent than using the normal equation.

Once you hav ethe gradient vector, which points uphill, just go in the opposite direction to go downhill.This means subtracting `   MSE` from `0`. This is where learning rate `n` comes into play.

Multiply the gradient vector by `n` to determine the size of downhill step

























If the learning rate is too low: the algorithm will eventually reach the solution, but it will take a long time. In the middle, the learning looks pretty good: In just a few iterations, it has already converged to the solution.

On the right, the learning rate is too high: the algorithm diverges, jumping all over the place and actually getting further and further away from the solution at every step.



#### Algorithm for batch gradient descent

























#### A quick implementation of this algorithm

















Hey, that's exactly ehat the normal equation fount! Gradient descent worked perfectly.

To find a good learning rate, you can use grid search. However, you may want to limit the number of iterations so that the grid search can eliminate models that take too long to converge.

If the number of iterations os too low, you will waste time while the model parameters do not change anymore. A simple solution is to set very large number of iterations but to inetrrupt the algorithm when the gradient vector becomes tiny - that is, when its norm becomes smaller than a tiny number (called the tolerance) - because this happens when gradient descent has (almost) reached the minimum.


### ADVANTAGES AND DISADVANTAGES OF BATCH GRADIENT DESCENT

#### Advantages of batch gradient descent
1. Fewer oscillations and noisy steps are taken towards the globola minima of the loss function because of updating the parameters by computing the average of all the training samples rather than the value of a singlr sample.
2. It can benefit from the vectorization which increases the speed of processing all training smaples all togather.
3. It produces a more stable gradient descent convergence and a stable error gradient than stochastic gradient descent.
4. It is computationally efficient as all computer resources are not being used to process all training samples.

#### Disadvantages of batch gradient descent
1. Sometimes a stable error gradient can lead to local minima and unlike stochastic gradient descent, no noisy steps are there to help get out of the local minima.
2. The  entire training set can betoo large to process in the memory due to which additional memory might be needed.
3. Depending on the computer resources it can take too long for processing all training samples as a batch.








### Stochastic gradient descent
Batch gradient descent uses the whole training set to compute the gradients at every step, which makes it very slow when the is large.

Stochastic gradient descent just picks up a random instance in the training set at evry step and computes the gradient based on anly on that single instance hence;
- SGD algorithms is much faster since it has very litte data to manipulate at evry situation.
- SGD algorithm works well with when the training on huge taining sets, since only one instance needs to be in memory at each iteration.

On the other hand, due to its stochastic9


















