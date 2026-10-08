# ElecAEOP
# Investigating Trainable Activation Functions for Neural Network Robustness

## Overview

This project investigates whether a neural network's activation functions can be adjusted to improve its robustness to noisy image data while keeping the network's learned weights fixed.

I developed this project as part of a research internship involving machine learning and optics, implementing the experiments in Python using TensorFlow, NumPy, OpenCV, and related tools.

## Research Question

Can trainable parameters within a ReLU-like activation function improve a neural network's performance on noisy images without retraining the network's underlying linear weights?

## Approach

I implemented a custom trainable activation function:

$$
f(x) = s \cdot ReLU(x - t)
$$

where:

- `s` is a trainable slope
- `t` is a trainable threshold

The experiments follow a two-stage approach:

1. Train a neural network on clean MNIST data and save its learned weights.
2. Freeze the Dense-layer weights and investigate whether adjusting the activation-function parameters can improve performance on noisy data.

This allows the experiments to focus specifically on the behavior of the nonlinear activation functions rather than retraining the entire network.

## Noise Experiments

The notebook includes experiments involving different forms of image distortion, including:

- Gaussian noise
- Salt noise
- Pepper noise
- Blur
- Gaussian blur
- Sharpening
- Laplacian filtering
- Edge detection

Noise parameters can also be varied to investigate how model performance changes as the severity of the distortion increases.

## Implementation

The project uses:

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pandas
- Matplotlib
- Jupyter Notebook

The notebook also includes custom training utilities for:

- Learning-rate scheduling
- Early stopping
- Training-history visualization
- Learning-rate tracking
- Tracking learned activation parameters
- CSV logging of experimental results
- Model weight inspection

## What I Explored

A major part of the project was investigating how much of a neural network needs to be retrained to adapt to noisy data.

Rather than modifying all of the learned weights, I explored whether changing only the parameters of the nonlinear activation functions could affect the network's response to noisy inputs.

The notebook contains multiple experiments and comparisons across different noise conditions and activation parameters.

## Repository Contents

- `TestingNonLF.ipynb` — Main notebook containing the implementation and experiments.


## Notes

This repository contains experimental research code rather than a production-ready machine-learning library. The notebook was developed to investigate a research question and explore different experimental configurations.

## Disclaimer 
The current code uses Gaussian noise for the active test case. The implemented functions support additional noise types, which can be tested individually at different times to allow simulations to run smoothly. Additional noise types and their parameters can also be specified manually.
