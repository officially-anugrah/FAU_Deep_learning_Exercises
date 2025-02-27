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

## Exercise 4: PyTorch for Classification
### Overview
This folder contains the implementation of **Exercise 4** from my deep learning coursework. The exercise focuses on implementing a **classification model using PyTorch**, specifically a version of the **ResNet architecture** to detect defects in solar cells.<br />
*Note: I have also kept the model which i have submitted for the challange server as we cannot modify the model architecture. if you want to see how far we can push and get the best results, check my other repo that i will be working on next. Most probably named Advanced solar cell crack detector.*

### Objectives
- Implement a **convolutional neural network (CNN)** based on ResNet.
- Utilize **PyTorch** to build and train the model.
- Handle **image-based classification** for detecting:
  - **Cracks** in solar cells.
  - **Inactive regions** in solar cells.
- Optimize hyperparameters to achieve the best **F1 score**.
- Compete in an **open classification challenge**, with results displayed on an online leaderboard.

### Dataset
Extract the images.zip file.
The dataset consists of electroluminescence images of solar cells, labeled with:
- **Crack presence** (binary classification: 0 or 1).
- **Inactive region presence** (binary classification: 0 or 1).

#### Data Format
- Images are in **PNG format**.
- Labels are provided in **data.csv**, which contains:
  - File path to each image.
  - Corresponding crack and inactive region labels.

### Implementation Details
- **Language:** Python
- **Libraries Used:** PyTorch, NumPy, Pandas, OpenCV
- **Key Components:**
  - **ResNet-based CNN** for feature extraction and classification.
  - **Data preprocessing pipeline** for image normalization and augmentation.
  - **Loss function and optimizer** setup for efficient training.
  - **Model evaluation** using accuracy, precision, recall, and F1 score.
  - **Hyperparameter tuning** for optimal performance.