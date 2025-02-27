# FAU Deep Learning Exercises
This repo contains all the exercises we have in our Deep learning course with its implemented code. <br />
*Note: Please first try to code by yourself and if you get stuck then you can use this repo as you reference. Thanks.*

## Author
**Anugrah Chimanekar**<br />
[LinkedIn](https://www.linkedin.com/in/anugrah-chimanekar/) - Feel free to reach out for feedback or suggestions!

---

## Exercise 0
### Overview
This folder contains the implementation of **Exercise 0** from my deep learning coursework. The purpose of this exercise is to refresh your knowledge of Python and NumPy. We will be implementing code for some simple tasks. <br />
*For more details check out the Exercise_0/Description.pdf*

## Exercise 1: From Scratch
### Overview
This folder contains the implementation of **Exercise 1** from my deep learning coursework. The exercise focuses on fundamental deep learning concepts, including forward and backward propagation, optimization techniques, and training simple neural networks.

### Objectives
- Implement a basic **fully connected neural network** from scratch.
- Understand **gradient descent** and how it updates model parameters.
- Train a model on a simple dataset and evaluate its performance.

### Implementation Details
- **Language:** Python
- **Libraries Used:** NumPy, Matplotlib (for visualization)
- **Key Components:**
  - Forward propagation
  - Loss computation
  - Backpropagation
  - Parameter updates using **Stochastic Gradient Descent (SGD)**

## Exercise 2: Convolutional Neural Networks
### Overview
This repository contains the implementation of **Exercise 2** from my deep learning coursework. The exercise focuses on extending the neural network framework to include building blocks for **Convolutional Neural Networks (CNNs)**, initialization schemes, advanced optimizers, and key CNN layers.

### Objectives
- Implement various **weight initialization schemes** (Constant, UniformRandom, Xavier, He).
- Develop advanced **optimization algorithms** (SGD with Momentum, Adam Optimizer).
- Implement essential **CNN layers**:
  - Convolutional Layer
  - Max-Pooling Layer
  - Flatten Layer
- Ensure compatibility between fully connected and convolutional layers.

### Implementation Details
- **Language:** Python
- **Libraries Used:** NumPy, SciPy (for convolution operations)
- **Key Components:**
  - Custom initialization strategies for network weights.
  - Forward and backward propagation for CNN layers.
  - Implementation of momentum-based and adaptive optimization algorithms.
  - Integration with existing neural network framework.

## Exercise 3: Regularization and the Recurrent Layer
### Overview
This folder contains the implementation of **Exercise 3** from my deep learning coursework. The exercise focuses on extending the neural network framework to include **regularization techniques** and **recurrent layers**.

### Objectives
- Implement **regularization strategies** to reduce overfitting:
  - L1 Regularization
  - L2 Regularization
- Refactor the framework to handle training and testing phases.
- Develop a **base optimizer class** to support regularizers.
- Extend the neural network to include **Recurrent Neural Network (RNN) layers**.

### Implementation Details
- **Language:** Python
- **Libraries Used:** NumPy
- **Key Components:**
  - Addition of a `testing_phase` boolean member in the `BaseLayer` class.
  - A `phase` property in `NeuralNetwork` to set the mode (train/test) for all layers.
  - A `BaseOptimizer` class to serve as a parent for optimization algorithms.
  - Implementation of `L1Regularizer` and `L2Regularizer` in `Constraints.py` for weight norm constraints.
  - Refactored optimizers to support regularization via `calculate_gradient(weights)`.
  - Modification of `NeuralNetwork` to add regularization loss to the data loss.