---
layout: post
title:  "Machine Learning : Introduction"
date:   2021-12-15 
category: tech
---

# Machine learning Defenition:
Machine learnig is a core research field of AI, and it is also necessary knowledge for deep learning. Therefore, this chapter mainly introduces the main concept of machine learning,and the common algorithms of machne learning


**Machine learning Algorithms(!)**

- Machine learning (including deep learning) is a study of the learning algorithms.A computer program is said to learn from experience E with respect to some class of task T and performance measure P, if its performance at tasks T measured by P, improves with experience E.


# **DIFFERENCES BETWEEN MACHINE LEARNING ALGORITHMS AND TRADITIONAL RULE-BASED ALGORITHMS**

- In rule based algorithms explicit programming is used to solve prooblems while in machine learning samples are used for training.

- Rules can be manually specified in rule based programming while in machile learning the decision making rules are complex or difficult to describe.

- In machine lerning rules are automatically learned by machines.



### Key Elements Of M.L 
There are tens of thousands of machine learning algorithms and hundreds of new algorithms are developed every year. Every machine learning algorithm has three components:
1. Representation: how to represent knowledge. Examples include decision trees, sets of rules, instances, graphical models, neural networks, support vector machines, model ensembles and others.
2. Evaluation: the way to evaluate candidate programs (hypotheses). Examples include accuracy, prediction and recall, squared error, likelihood, posterior probability, cost, margin, entropy k-L divergence and others.
3. Optimization: the way candidate programs are generated known as the search process. For example combinatorial optimization, convex optimization, constrained optimization.
All machine learning algorithms are combinations of these three components. A framework for understanding all algorithms.






# Application scenarios of machine learning
1. The solution to a problem is complex, or the problem may involve a large amount of data without a clear data distribution function.

2. Machine learning can be used in the following scenarios:
- Rules are complex or cannot be described, such as voice recognition.

- Task rules change over time for example, in the part of speech tgging tasks, new words or meanings are generated at any time.

- Data distributio changes over time, requiring constant readaptation of programs such as predicting the trend of commodity sales.

# **RATIONAL UNDERSTANDING OF MACHINE LEARNING ALGORITHMS**

- Target function f is unknown. Learning algorithms cannot obtain a perfect function f.
- Assume that hypothesis function g approximates function f, but may be different from function f.


**Main problems solved by machine learning**

#### Machine learning can deal with many types of tasks. The following describes the most typical and common types of tasks:

- Classification: A computer program needs to specify which of the k categories some input belongs to. To acomplish this task, lerning algorithm usually output a function() for example, the image classification algorithm computer vision is designed to handle classification tasks.

- Regression: for this type of task, a computer program predicts the output for the given input larning algorithm, typically output a function () An example of this task type is to predict the claim amount of an insured person (to set the insuarence premium) or predict the security price.

- Clustering: A large amount of data from an unlabeled dataset is divided into multiple categories according to internal and similarity of the data. Data in the same category is more similar than that in different categories. This feature can be used in scenarios such as image retrieval and user profile management.

- ***Classificatin and regression are the two main types of prediction, accounting from 80% to 90%. The output of classification is descrete category values, and the output of regression is continous numbers.***

## Machine Learning Classification
1. **Supervised learning:** Obtain an optimal model with required performance through training and learning based on the samples of known categories. Then, use the model to map all inputs and check the output for the purpose of classifying unknown data.

2. **Unsupervised learning:** For unlabeled samples, the learning algorithms directly model the input datasets. Clustering is a common form of unsupervised learning. We only need to put highly similar samples togather, calculate the similarity between ne samples and existing ones, and classify them by similarity.

3. **Semi-supervised learning:** In one task, a machine learning model that automatically uses a large amount of unlabeled data to assist learning directly of small amounts of labeled data.

4. **Reinforcement learning** It is an area of machine learning concerned with how agents ought to take actions in an environment to maximize some notion of cumulative reward. The difference between reinforcement learning and supervised learning is the teacher signal. The reinforcement signal provided by the environment in reinforcement learning is used to evaluate the action (Scalar signal) rather than tellingthe learning system how to perform correct actions.

- Reinforcement learning is a very different beast. The learning system, called **agent** inthis context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative reward).
- It must then learn by itself what is the best strategy, called a **policy**, to get most reward over time.

- Apolicy defines what ***action*** the agent should choose when it is in a given situation.

- The model perceives the environment, takes actions, and makes adjustments and choices based on the status and reward or punishment.

Reinforcement learning always looks for the best behaviours. Reinforcement learning is targeted at machines or robots.
 - Autopilots: Should it break or accelerate when the yellow light stsrts to flash?
 - Cleaning robots: Should it keep working or go back for charging.

 ### Example Applications of Reinforcement Learning project.

- DeepMind's AlphaGo program: It made headlines in March 2016 when it beat the world champion Lee Sedol at the game of Go.
- It learned its winning policy by analyzing millions of games, and then playing many games against itself.
- Note that learning was turned off during the game against the champion; AlphaGo was jjust applying the policy it had learned.


## Batch Vs Online Learning
Another criterion used to classify Machine Learning systems is wheather or not the system can learn incrementallyfrom a stream of incoming data.

**Batch learning also called offline learning**
- In batch learning, the system is incapable of learnig incrementally: It must be trained using all the available data.
- This will generally take a lot of time and computing resources, so it is typically done offline.
- First the system is trained, and then it is launched into production and runs without learning anymore; It just applies what it has learned.
- If you want a batch learning system to know about new data(such as new type of spam), you need to train a new version of the system from scratch on the full dataset(not just the new data, but also the old data), then stop the old system and replace it with the new one.
- Training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data(e.g. to predictstock prices), then you need a more reactive solution.
- Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.


**Online Learning**
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini batches.

- Each learning step is fast and cheap, so the system can learn about the new data on the fly, as it arrives.
- Online learning is great for systems that receives data continously flow (e.g., Stock prices) and need to adapt to change rapidly and autonomously.

- It is also good option if you have limited computing resources: Once online learning system has learned about new data instances, it does not need them anymore, so you can discard them(unless you want to roll back to a previous state and "replay" the data) This can save a huge amount of space.

- Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine's main memory(This is called out-of-core learning).



## **Instance based Vs Model-Based Learning**
One more way to categorize ML system is by how they generalize.
Most ML tasks are about  making predictions. This means that a given number of training examples, the system needs to be able to generalize to example it has seen before.

- Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances.
- There are two main approaches to generalization: Instance based and Model based learning.


**Instance-based Learning**
Possibly the most trivial form of learning is simply to learn by heart. If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users - not the worst solution but certainly not the best.

Instead of just flagging emails that are identical to known spam emails you spam filter could be programmed to also flag emails that are very similar to known spam emails. This requires a measure of similarity between two emails.

A(Very basic) similarity measure between two emails could be to count the number of words they have in common. The system wouldfag an emailas spam if it has many words in common with a known spam email.

This is called instance-based learning: The system learns the examples by heart, then generalizes to new cases using a similarity measure.




**Model-Based Learning**
Model based learning algorithms use the taining data to create a model that has parameters learned from the training data. 

For Example: In Support Vector Machines ***(SVM)*** We have

W * (LEARNED WEIGHTS VALUE) And b * (LEARNED BIAS VALUE)

After the model is built, the training data can be discarded

**Instance based Learning**
Instance-based learnig algorithms use the entire dataset as the model. For example: K-Nearest Neighbors **(KNN)** Algorithms looks at the close neighborhood of the input example in the space of feature vectors and outputs the abel that it saw the most often in this close neighborhood.



## ***Which is better ?***

- Most supervised learning algorithms are model-based. In Model-based you can generalize your rules in the form of a model which can be stored whereas in instance-based, generalization happens for each scoring instance individually as when seen. 
-Scoring for new instances is faster in model-based than instance-based and size of model stored is less than storing training data. Data cannot be discarded in former while it is possible in letter.


Model training is often the biggest and tmme-consuming component of an analytics project. So, what does a lazy scientist do ?

Instance-based learnig or lazy learning doesn't build the model a-priori. It only queries the training data on demand during scoring for each specific new instnce.

So if this is what you are looking for, Instance-based learning might be for you. For the best, modle-based learning is the wayto go with.

## **ADVANTAGES OF MACHINE LEARNING**

1. **Easily identifieshidden trends and patters**
- Mahine learning can review large volumes of data and discover specific trends and patterns that would not be apparent to humans.

2. **Continous Improvements**
- We are continously generating new data and when we provide this data to the machine learning model which helps it to upgrade with time and increases its performance and accuracy. We can say it is like gaining experience as they keep improving in accuracy and efficiency. This lets them make better decisions.

3. **Handling multidimensional and multi-variety data complex for human**
- Machine learning algorithims are good at handling data that are multidimensional and multi-variety, and they can do this in a dynamic or uncertain environments.

4. **Wide applications**
- Machine learning is used in wide decision making application in where it does not appl, it holds the capability to hep deliver a much more personal experience to customer while also targeting the rightcustomers..




## **DISADVANTAGES OF MACHINE LEARNING**

1. **Data Acquisition**
- Machine learning requires a massive amount of data sets to train on, and these should be inclusive/unbiased, and of good quality.

2. **Time and resources**
- Machine learning needs enough time to let the algorithms learn and develop enough to fulfil their purpose with a conciderable amount of accuracy and relevancy. It also needs massive resources to function. This can mean additional requirements of computer power for you.

3. **Interpretation of results**
- Another major challenge is the ability to accurately interpret results generated by the algorithms for your purpose. Sometimes, based on some analysis you might select an algorithm but it is not necessary that the model is the best for the problem.

4. **High-Error-Susceptibility**
- Machine learning is autonomous but highly susceptible to errors. Suppose you train an algorithm with datasets small enough to not bee inclusive. You end up with biased predictions coming from a biased training set.

5. **Curse of dimensionality**
- Another problem Machine learning model faces is too many feature sof data points. This can be a real hindrance.

6. **Difficulty in deployment**
- Complexity of the ML model makes it quite difficult to be deployed in real life.




## **APPLICATIONS OF MACHINE LEARNIG**
- Automatic language translation
- Medical diagnostics
- Stock market trading
- Online fraud detection
- Virtual personal assistant
- Email spam and malware filtering
- Self driving cars
- Product recommendations
- Traffic prediction
- Speech recognition
- Image recognition


## **HISTORY OF MACHINE LEARNING**

***The early history of Machine learning (pre-1940)***

**1834:** In 1834, Charles Babbage, the father of computing, conceived a device that could be programmed with puch cards. Howwever the machine was never build, but all modern computers rely on this logic structure.

**1936** In 1936, Alan Turing gave a theory that how a machine can determine and execute a set of instructions.

**The era of a stored program computers**
**1940:** In 1940, the first manually operated computer, "ENIAC" was invented.

**1943:** In 1943 a human neural network was modeled with an electrical circuit. In 1950, the scientists started applying their idea to work and analyzed how huan neurons might work.


**Computer Machhinery and intelligence**
**1950:** In 1950,Alan Turing published a seminal paper, "Computer machinery and intelligence", on the topic of artificial intelligence in his paper he asked, "Can machines think?"


**Machie Intelligence and games**
**1952:** Arthur Samuel, who was the pioneer of machine learning, created a program that helped an IBM computer to play a checkers game. It performed better more it played.

**1959:**   In 1959 The term "Machine Learning" was first coined by Arthur Samuel.

**The first AI Winter:** 
- The duration of 1974 to 1980 was tough time for AI and ML resesrchers, and this duration was called AI winter. In this duration failure of machine translation occured, and people had reduced their intrest in AI, which led to reduced funding by the government on researchers.

**Machine learning fro theory to reality**
**1959:** The first Neural network was applied to a real-world problem to remove echoesover phone lines using an adaptive filter.

**1985:** Terry Sejowski and Charles Rosenberg invented a neural network NETtalk, which was able to teach itself how to correctly pronounce 20,000 words in one week.

**1997:** The IMB's DEEP BLUE intelligent computer won the chess game aginst the chess expert Garry Kasparov, and it became the first computer which had beaten a human expert.

**Machien learning at 21st century**
**2006:** In 2006, computer scientists Geoffery Hinton has given a new name to neural net research as "DEEP LEARNING", and nowadays, it has become one of the most trending technologies.

**2012:** Google created a deep neural networ which learned to recognize the image of humans and cats in youtube videos.

**2014:** In 2014, the chatbot "Eugine Gootsman" cleared the Turing test. It was the first chatbot who convinced the 33% of human judges that it was not a machine.

**2014:** DeepFace was a deep neural network craeted by facebook, and they claimed that it could recognize a person with the same precision as a human can do.

**2016:** AlphaGo beat the world's number second player Lee Sedol at GO game: In 2017 it beat the number one player of this game Ke Jie.

**2017** Alphabet's jigsaw team built an intelligent system that was able to learn the online trolling. It used to read millions of comments of different websites to learn to stop online trolling.

**Machine Learning at present:**
Now Machine learning has got a great advancements in its research, and it is present everywhere around us, such as self driving cars, amazon Alexa, Catboats, recommender systems and many more. It includes supervised ,unsupervised and reinforcement learning with clustering , classification, decision trees, SVM algorithms, etc.

Modern machine learning models can be used for making various predictions, including disease prediction, weather prediction, stock market analysis, etc.





 







