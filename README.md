# 🧠 AI/ML Definitions and Explanations

1. What is Artificial Intelligence?
   Artificial Intelligence is a set of technologies, those make the computers enable to see, understand and translate spoken or written languages, recommendation systems and make some intelligence predictions on data.
   In other words, Artifical Intelligence is a field of study, where we try to make computers smarter. It's mainly mimicking human intelligence.

2. What is Machine Learning?
   Machine Learning is a subset of Artificial Intelligence. We train the systems with the data, then systems makes some intelligence predictions on data without being explictly programmed.
   In other words, Machine Learning (ML) is finding patterns in the data, and use those patterns to predict or classify the things.

3. How many types of Machine Learning are there? Explain them?
   There are mainly three types of Machine Learnings are there:
   They are:
   1. Supervised Learning:
      Supervised Learning is a type of Machine Learning. Where we train the system on the labeled data.
      In other words, Supervised Learning is a type of Machine Learning. where we get a label with the output.
      
   3. Unsupervised Learning:
      It's a type of Machine Learning. In this Learning we train the systems on unlabeled data.
      In other words, Unsupervised Learning is a type of Machine Learning. Where we get only the output without the label
      
   3. Reinforcement Learning:
      Reinforcement Learning is also one type of Machine Learning. It is quit differnet when we compare it with the supervised and unsupervised learning. In this method, there is an agent and a environment is there, whether the agent perform well, it get the rewards other wise it get punishment.

4.  Explain Semi-Supervised Learning?
    Machine Learning is broadly classified into three types. Supervised, Unsupervised and Reinforcement Learning. Semi-supervised learning is also a type of Machine Learning, it falls between supervised learning and unsupervised learning. It suit best when the training data is huge and it's unlabeled, some way labeling is not happening. In those cases we can use semi-supervised learning where we train the model with the small amount of labeled data and large amount of unlabeled data.

5. What is Deep Learning?
   Deep Learning is a subset of Machine Learning. Where 
7. 📚 What is TensorFlow?
   
   TensorFlow is an open-source end-to-end machine learning library for preprocessing data, modeling data, and serving models (getting them into the hands of others).

8. 🤔 Why Use TensorFlow?

   Rather than building machine learning and deep learning models from scratch, it's more likely you'll use a library such as TensorFlow. This is because it contains many of the most common machine learning functions inbuilt you'll want to use.

3. 
   
2. What is Machine Learning?
3. Can you explain types of Machine Learning?
4. What is Deep Learning?
5. What is a neuron?
6. Explain briefly about input layer, hidden layer and output layer?
7. What is shallow neural network?
8. what is an optimizer? why do we use it?
9. what is activation function?
10. Types of ativation funtions?
11. what is loss functions?
12. How do we evaluate a model?
13. What is SGD(Stochastic Gradient Descent)?
14. What is Adam?
15. What is MAE(Mean Absolute Error)?
16. What is MSE(Mean Squared Error)?
17. what is accuracy? which we used it in evaluating metrics for Classification model?
18. What is Confussion Metrics?




3. 🔢 What is a Tensor?

A Tensor is an N-dimensional matrix where the matrix could be almost anything you can imagine. It is a generalization of vectors and matrices to higher dimensions.

It could be numbers themselves (using tensors to represent the price of houses).
It could be an image (using tensors to represent the pixels of an image).
It could be text (using tensors to represent words).
Or it could be some other form of information (or data) you want to represent with numbers.
If you've ever used NumPy, tensors are kind of like NumPy arrays. The main difference between tensors and NumPy arrays is that tensors can be used on GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units).

4. 🔢 What is a Tensor Rank?
   
The number of directions a tensor can have in an N-dimensional space is called the "Rank" of the tensor.

The rank is denoted R.

> A Scalar is a single number.

* It has 0 axes.
* It has a rank of 0.
* It's a 0-dimensional tensor.

> A Vector is an array of numbers.

* It has 1 axis.
* It has a rank of 1.
* It's a 1-dimensional tensor.

> A Matrix is a 2-dimensional array.

* It has 2 axes.
* It has a rank of 2.
* It's a 2-dimensional tensor.

> Real Tensor

Technically, all the above are tensors (meaning scalar, vector, and matrix), but when we speak of tensors, we generally speak of matrices with a dimension larger than 2 (R > 2).

5. 📐 What is the Shape, Rank, and Size of a Tensor?
   
At the time of model building or preprocessing data, most of the time is spent matching the mismatched tensor shapes. So you'll want to get different pieces of information from the tensors. In particular, you should know the following tensor vocabulary:

* **Shape**: The length (number of elements) of each of the dimensions of a tensor.
* **Rank**: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix has rank 2, and a tensor has rank n.
* **Axis** or **Dimension**: A particular dimension of a tensor.
* **Size**: The total number of items in the tensor.
