---
layout: post
title: Introduction to Machine learning 
subtitle: Notes on Deep learning with Python book
categories: machinelearning
---
# Four branches of machine learning

## Supervised learning
It consists of learning to map input data to known targets:
$f: x (input) \rightarrow y (output)$
There are 2 major techniques in supervised: classification and regression, and there are more exotic variants as well:

* Sequence generation, eg: captioning the picture
* Syntax tree prediction, eg: decomposition a sentence
* Object detection, eg: draw a bounding box aroung certain objects inside a picture
* Image segmentation, eg:..

## Unsupervised learning
It consists of finding interesting transformations of the input data without targets.
Application of unsupervised learning is supports in data visualization, data compression or data denoising or to better understanding the correlation present in the data.
Well-known categories of unsupervised learning are **Dimensionality reduction** and **clustering**

## Self-supervised learning
It is supervised learning without human-annotated labels, they're generated from the input data, typically using a heuristic algorithms

Well-knowns instance of self-supervised learning: autoencoders

## Reinforcement learning
An *agent* receives information about its environment and learns to choose actions that will maximize some reward. Some potential application area of reinforcement learning: self-driving cars, robotics, resource management, education,...

# Evaluating machine learning models
In machine learning, the main goal is to archive models that *generalize* (perform well on *never-before-seen* data) and *overfitting* is the central obstacle. There are some strategies to measure generalization or evaluate machine-learning models.

## Training, validation and test sets
The data set is divided into **THREE** sets:

* Training set: train the model
* Validation set: evaluate the model
* Test set: test the model

The model developing always involves tuning its configuration, eg:
* The number of layers
* The size of layers
* other *hyperparameters*...

Tuning the configuration can result in the *overfitting to the validation set* because of *information leaks*. Every time you tune a hyperparameter of the model based on the model's performance on the validation set, some information about the validation data leaks into the model. At the end, the model performs well on the validation data, so you need to use a completely differet dataset to evaluate the model: the test dataset.

There are some methods for validation:

### Simple hold-out validation

![](https://image.ibb.co/f6f1my/hold_out_validation.png)

### K-fold validation

![](https://image.ibb.co/bUm7Ry/k_fold_validation.png)

### Iterated K-fold validation with shuffling

It consists of applying K-fold validation multiple times and shuffling the data every time before splitting it K ways.

### Notes

* Data representativeness: you should randomly shuffle your data before splitting it to make sure that the training set and test set could be representative of the whole data.
* Time series data should not randomly shuffle, the test set should be posterior to the data in the training set.
* Redundancy in your data

# Data preprocessing, feature engineering, and feature learning

## Data preprocessing

Aim: making the raw data more amenable (easy to control) to neural networks.

### Vectorization

Inputs or targets in a neural network must be tensors of floating-point data (or integers).
If your data is sound, images, text, you should first turn into tensors. This step called *data vectorization*.

### Value normalization

In general, it isn't safe to feed the neural networks data that takes relatively large values or data the is heterogeneous (one feature is in the range 0-1, one feature is in the range 100-200).
This causes large gradient updates that will prevent the network from converging. To make learning easier for your network, your data should have the following characteristics:
* Take small values
* Be homogenous

The normalization (to mean of 0 and standard deviation of 1) is the common practice.

### Handling the missing values

In general, with neural networks, it's safe to input missing values as 0, with the condition that 0 isn't already a meaningful value. The network will learn from exposure to the data that the value 0 means missing data and will start ignoring the value.

## Feature engineering

The essense of feature engineering is making a problem easier by expressing it in a simpler way. It usually requires understanding the problem in depth.

Example:
Raw data: Pixel gird of clock -> Better feature: clock hands' coordinates {x1, y2}, {x2, y2} -> Even better features: angles of clock hands theta1, theta 2

Good features allow you to solve problems more elegantly while using fewer resources. For instance, it would be ridiculous to solve the problem of reading a clock face using a convolutional neural network.
Good features let you solve a problem with far less data. The ability of deep

# Overfitting and Underfitting

The fundamental issue in machine learning is the tension between optimization and generalization.
* Optimization: adjusting a model to get the best performance possible on the training data.
* Generalization: how well the trained model performs on data it has never seen before.

At the beginning of training: optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening, your model is said to be **underfit** -> there is still progress to be made; the network hasn't yet modeled all relevant patterns in the training data.

After a certain number of iterations on the training data, generalization stops improving and validation metrics stall and then begin to degrade: the model is starting to **overfit** -> it's beginning to learn patterns that are specific to the training data but that are misleading or irrelevant when it comes to new data.

**Get more training data is usually useful:** a model trained on more data will naturally generalize better.
If more data failed, the next solution is **to modulate the quantity of information that your model is allowed to store or to add constraints on what information it's allowed to store.** -> **regularization**.

## Reducing the network's size
This is the simplest way to prevent overfitting.

The network's size or The number of learnable paramaters in a model (determined by the number of layers and the number of units per layer) is often referred to as the model's capacity. A model with more parameters has more memorization capacity, so it can easily learn a perfect dictionary-like mapping between training samples and their targets - a mapping with out any generalization power.

Deep learning models tend to be good at fitting to the training data, but the real challenge is generalization.

The model should have just enough parameters that they don't underfitting.

**No formula to determine the right number of layers or the right size of each layer**. Need to research to find the optimal model size of your data.

## Adding weight regularization

This way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights to take only small values, which makes the distribution of weight values more **regular** => **weight regularization**
There are 2 flavors:
* L1 regularization -> The cost added is proportional to the *absolute value of the weight coefficients* (L1 norm of the weights)
* L2 regularization -> The cost added is proportional to the *square of the value of the weight coefficients* (L2 norm of the weights/ *Weight decay*)

## Adding dropout
**Dropout** is one of the most effective and most commonly used regularization techniques for neural networks.
When apply to a layer,dropout consists of randomly dropping out (setting to zero) a number of output features of the layer during training.

Eg: [0.2, 0.5, 1.3, 0.8, 1.1] --dropout--> [0, 0.5, 1.3, 0, 1.1]

The **dropout rate** is the fraction of the features that are zeroed out; it's usually set between 0.2 and 0.5.

# The universal workflow of machine learning
## Defining the problem and assembling a dataset
* What will your input data be?
* What are you trying to predict?
* What type of problem are you facing?
* Is it binary classification or multiclass classification? Scala or vector regression? Multiclass, multilabel classification?...

## Choosing a measure of success
For regression problems:
* Accuracy? Precision and Recall?
* Customer-retention rate?

For balanced-classification problems, accuracy and *area under the receiver operating chracteristic curve (ROC AUC)* are common metrics.

For ranking problems or multilabel classification: mean average precision.
## Deciding on an evaluation protocol
* Maintaining a hold-out validation set: having plenty of data
* Doing K-fold cross-validation: having too few samples for hold-out validation to be reliable
* Doing iterated K-fold validation: for performing highly accurate model evaluation when little data is available.

## Preparing your data
* Formatted as tensors
* Scaled to small values
* Normalized
* Feature engineering

## Developing a model that does better than a baseline
* Last-layer activation: eg: sigmoid in the last layer, regression didn't use last-layer activation,...
* Loss function
* Optimization configuration: which optimizer?

| Problem type                            | Last-layer activation | Loss function              |
|-----------------------------------------|-----------------------|----------------------------|
| Binary classification                   | sigmoid               | binary crossentropy        |
| Multiclass, single-label classification | softmax               | categorical crossentropy   |
| Multiclass, multilabel classification   | sigmoid               | binary crossentropy        |
| Regression to arbitrary values          | None                  | MSE                        |
| Regression to values between 0 and 1    | sigmoid               | MSE or binary crossentropy |
Regression to values between 0 and 1|sigmoid| MSE or binary crossentropy

## Scaling up: developing a model that overfits
* Add layers
* Make the layers bigger
* Train for more epochs

→ Monitor the training loss and validation loss or other metrics.

## Regularizing your model and tuning your hyperparameters
* Add dropout
* Try different architectures: add or remove layers
* Add L1 and or L2 regularization
* Try different hyperparameters to find the optimal configuration
* Optionally, iterate on feature engineering: add new features, remove features.
