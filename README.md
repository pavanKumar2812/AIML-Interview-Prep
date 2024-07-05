# 🧠 AI/ML Definitions and Explanations
============================================

📚 What is TensorFlow?

TensorFlow is an open-source end-to-end machine learning library for preprocessing data, modeling data, and serving models (getting them into the hands of others).

🤔 Why Use TensorFlow?

Rather than building machine learning and deep learning models from scratch, it's more likely you'll use a library such as TensorFlow. This is because it contains many of the most common machine learning functions inbuilt you'll want to use.

🔢 What is a Tensor?

A Tensor is an N-dimensional matrix where the matrix could be almost anything you can imagine. It is a generalization of vectors and matrices to higher dimensions.

It could be numbers themselves (using tensors to represent the price of houses).
It could be an image (using tensors to represent the pixels of an image).
It could be text (using tensors to represent words).
Or it could be some other form of information (or data) you want to represent with numbers.
If you've ever used NumPy, tensors are kind of like NumPy arrays. The main difference between tensors and NumPy arrays is that tensors can be used on GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units).

##🔢 What is a Tensor Rank?
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

## 📐 What is the Shape, Rank, and Size of a Tensor?
At the time of model building or preprocessing data, most of the time is spent matching the mismatched tensor shapes. So you'll want to get different pieces of information from the tensors. In particular, you should know the following tensor vocabulary:

* **Shape**: The length (number of elements) of each of the dimensions of a tensor.
* **Rank**: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix has rank 2, and a tensor has rank n.
* **Axis** or **Dimension**: A particular dimension of a tensor.
* **Size**: The total number of items in the tensor.
