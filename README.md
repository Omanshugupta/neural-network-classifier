# Neural Network Classifier – A, B, C (5x6 Grid)

This project implements a simple neural network from scratch using NumPy to classify binary images of letters A, B, and C represented as 5×6 (30-pixel) grids. No external machine learning libraries such as TensorFlow or PyTorch are used.

## Problem Statement

Build a feedforward neural network to classify 6×6 binary pixel patterns as either A, B, or C. Each image is a flattened 1D array of 35 binary values. The network must be trained using backpropagation and gradient descent.

## Project Structure

neural-network-classifier/

├── neural_network_classifier.ipynb

├── README.md

## Dataset
- No external dataset was used.
- Letters A, B, and C are defined manually as 5×6 binary arrays.
- Each letter is flattened to a 30-length 1D array.
- Labels: A = 0, B = 1, C = 2
- Labels are one-hot encoded for training.

## Model Architecture

- Input Layer: 30 neurons (flattened 5×6 grid)
- Hidden Layer: 12 neurons (with sigmoid activation)
- Output Layer: 3 neurons (softmax activation)

## Activation & Loss Functions

- Hidden Layer: Sigmoid
- Output Layer: Softmax
- Loss Function: Cross-entropy
- Optimization: Manual Gradient Descent

## Training Details

- Epochs: 1000 (configurable)
- Learning Rate: 0.1
- Loss tracking: Plotted over epochs
- Accuracy: Evaluated post-training by comparing predicted class with actual class

## Output & Visualization

- Loss vs. Epoch plot to monitor learning
- Displays predictions for each input pattern using `matplotlib.pyplot.imshow()` for visualization

## Results

- The network correctly classifies A, B, and C with high accuracy after training.
- Loss decreases steadily, confirming effective learning.

## Author

- Omanshu Gupta
- Full Stack Data Science & AI

## Tools Used

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook

## Submission Instructions

This repository is part of Assignment 6: Neural Network.
To run the notebook:
1. Open `neural_network_classifier.ipynb`
2. Run all cells in order (make sure required packages are installed)


