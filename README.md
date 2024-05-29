
!(/assets/images/architecture.png)

# Overview
This repository contains an implementation of a custom neural network model using PyTorch. The model architecture is designed to explore advanced concepts in recurrent neural networks and aims to provide a flexible framework for experimenting with different configurations.

# Model Architecture
The core of the model is the Branch_Cell, which consists of multiple LSTM-like units called "axis units" and a "DB compiler layer". Each axis unit contains the typical LSTM gates (input, forget, cell, output), and the DB compiler layer integrates information from all axis units to produce the final output.

# Components
# Branch_Cell

Input Size: Size of the input feature vector.
Hidden Size: Number of hidden units in the axis units.
Ahidden Size: Number of hidden units in the DB compiler layer.
Naxis: Number of axis units.
Device: Device on which the model is running (CPU or GPU).
# V_2

Layers: Number of layers of Branch_Cell.
Integrates multiple Branch_Cell instances and processes the input through these layers sequentially.
Detailed Description
Branch_Cell Class
The Branch_Cell class defines the structure of each axis unit and the DB compiler layer. The weights for each gate in the axis units and DB compiler layer are initialized using Xavier initialization and zeros for biases.

weights_conf(): Configures and returns the weights for the axis units and DB compiler layer.
init_weights(): Initializes the weights using Xavier uniform distribution.
diff_percent(): Calculates the difference percentage used in the forget gate.
forward(): Defines the forward pass through the Branch_Cell, updating the hidden and cell states.
V_2 Class
The V_2 class is a higher-level abstraction that stacks multiple Branch_Cell layers.

forward(): Defines the forward pass through all layers of Branch_Cell, sequentially updating the hidden and cell states and finally returning the output.
Usage
To use the model, instantiate the V_2 class with the desired parameters and pass the input tensor through it.

python

import torch

# Define device
device = torch.device('cpu')

# Instantiate the model
model = V_2(input_size=5, hidden_size=10, ahidden_size=20, naxis=4, layers=5, device=device)

# Create a random input tensor
X = torch.rand(30, 5, dtype=torch.float32)

# Forward pass
output = model(X)
print(output)
Requirements
torch
numpy
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/experimental-model.git
Install the required packages:
bash
Copy code
pip install torch numpy
