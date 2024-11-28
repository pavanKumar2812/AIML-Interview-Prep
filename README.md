# 🧠 AI/ML Definitions and Explanations

1. What is Artificial Intelligence?

   Artificial Intelligence is a set of technologies, those make the computers enable to see, understand and translate spoken or written languages, recommendation systems and make some intelligence predictions on data.
   In other words, Artifical Intelligence is a field of study, where we try to make computers smarter. It's mainly mimicking human intelligence.

3. What is Machine Learning?

   Machine Learning is a subset of Artificial Intelligence. We train the systems with the data, then systems makes some intelligence predictions on data without being explictly programmed.
   In other words, Machine Learning (ML) is finding patterns in the data, and use those patterns to predict or classify the things.

5. How many types of Machine Learning are there? Explain them?

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

7.  Explain Semi-Supervised Learning?

    Machine Learning is broadly classified into three types. Supervised, Unsupervised and Reinforcement Learning. Semi-supervised learning is also a type of Machine Learning, it falls between supervised learning and unsupervised learning. It suit best when the training data is huge and it's unlabeled, some way labeling is not happening. In those cases we can use semi-supervised learning where we train the model with the small amount of labeled data and large amount of unlabeled data.

9. What is Deep Learning?

   Deep Learning is a subset of Machine Learning. It uses multilayered neural network called Deep Neural Network, where we use them to similate the decision-making power of human brain. Some form of the deep learning powers most of the Artificial Intelligence application in our day to day life.

10. What is Artifical Neural Network?

    Artificial Neural Network (ANN) is a deep learning algorithm, where it is inspired by the structure and function of the Biological Neural Network of human brain. It is an approch on mimic the human intelligence, it woking is same as the biological neural network, but doesn't exactly resemble its working.  

    Artificial Neural Network (ANN) would only accepts the numerical or structured data as input. To accept the non-numerical or unstructured data formats such as Image, Text, Speech, Convolutional Neural Network (CNN) or Recursice Neural Network (RNN) are used respectively.

12. What is Neuron?

    In the field of Artificial Intelligence, A Neuron in a neural network is a fundamental unit of processing and produce an output.

13. What is shallow neural network?
    
    A shallow neural network is a neural network, it consist only one hidden layer in between the input and output layers.

14. What is a node in Neural Network?

    A Node in Neural Network is a computational unit that performs a weighted sum of its input, and applies a activation function to its sum, and produce an output, that can be passed to the subsequent nodes or used as a final result. Nodes are organised in a layers essitential for finding complex patterns in data.

15. Explain briefly about input layer, output and hidden layers?

    An ANN (Artificial Neural Network) is consist the similar structure and functional of the biological neural system of human brain. But, doesn't exactly resemble its working. It consist of input, output and n-hidden layers.

    In Neural Network a layer is a collection of nodes together it performs the data. Layers are organized in sequentially, and each type of the layer serves specific purpose.
     
    **Input Layer:** The first layer, which receives the raw data. Each node in this layer corresponds to a feature or variable in the data.
    **Hidden Layer:** These layers are come after the input layer and before the output layer. Hidden layers can vary in number and size, and nodes apply weights and bias, activation functions to the inputs they received.
    **Output Layer:** This is the final layer, which produces the networks prediction or result. Each node in this layer corresponds to a different output or class, based on the problem. (Example: Classification or Regression).

16. 🔢 What is a Tensor?

   A Tensor is an N-dimensional matrix where the matrix could be almost anything you can imagine. It is a generalization of vectors and matrices to higher dimensions.
   
   It could be numbers themselves (using tensors to represent the price of houses).
   It could be an image (using tensors to represent the pixels of an image).
   It could be text (using tensors to represent words).
   Or it could be some other form of information (or data) you want to represent with numbers.
   If you've ever used NumPy, tensors are kind of like NumPy arrays. The main difference between tensors and NumPy arrays is that tensors can be used on GPUs (Graphical Processing Units) and TPUs (Tensor Processing Units).

17. 🔢 What is a Tensor Rank?
   
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

18. 📐 What is the Shape, Rank, and Size of a Tensor?
   
   At the time of model building or preprocessing data, most of the time is spent matching the mismatched tensor shapes. So you'll want to get different pieces of information from the tensors. In particular, you should know the following tensor vocabulary:
   
   * **Shape**: The length (number of elements) of each of the dimensions of a tensor.
   * **Rank**: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix has rank 2, and a tensor has rank n.
   * **Axis** or **Dimension**: A particular dimension of a tensor.
   * **Size**: The total number of items in the tensor.
    
19. 📚 What is TensorFlow?
   
   TensorFlow is an open-source end-to-end machine learning library for preprocessing data, modeling data, and serving models (getting them into the hands of others).

20. 🤔 Why Use TensorFlow?

   Rather than building machine learning and deep learning models from scratch, it's more likely you'll use a library such as TensorFlow. This is because it contains many of the most common machine learning functions inbuilt you'll want to use.

21. What is Bias-Variance trade-off?
    Bias-Variance trade-off is finding the sweet spot between the overfitting and the underfitting
    
    Bias is the difference between the average prediction of our model and the true values which we are trying to predict. Model with the high bias pay very little attention to the training data and oversimplifies the model. This leads to high error on training and test data.
    Vaiance is the difference in the fit of the datasets is called variance.
    *High Variance - Overfitting*
    *High Bias - Underfitting*

22. How is KNN different from a K-mean?
    At first, it may seems similar but it's not. K-Nearest algorithm is a supervised classification algorithm while K-mean algorithm is a unsupervised clustering algorithm, while the mechanism may seen similar at first using K-Points, they are totally different algorithms.
    In order to work with K-Nearest Neighbour algorithm we need labelled data for the unlabelled points. K-Mean clustering algorithm requires only a set of unlabelled points and a threshold, the algorithm will take unlabelled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.
    The critical difference here is that KNN needs labelled data and thus it is a supervised learning, while k-mean doesn't need labelled data it's a unsupervised learning.

23. How do you implement the K-mean algorithm?
    * Specify number of clusters K
    * Initialize centroid by first shuffling the dataset and then randomly selecting K data points for the centroid without replacement.
    * Keep iterating until there is no change to the centroids i, e assignment of the data points to the clusters isn't changing.
      * Compute the sum of the squared distance between points and all the centroids.
      * Assign each data point to the closest cluster(centroid).
      * Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
     
24. How to 
      
    




