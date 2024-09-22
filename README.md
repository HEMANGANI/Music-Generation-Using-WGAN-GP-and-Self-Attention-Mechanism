# From Bytes to Beats: Music Generation Using WGAN-GP and Self-Attention Mechanism

This repository contains the implementation of a music generation model using Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP), enhanced with a self-attention mechanism designed to create melodic compositions. This project aims to explore the capabilities of GANs in the realm of music generation. Utilizing the 17-track Lakh Pianoroll Dataset, the model applies advanced techniques to generate high-quality and coherent musical sequences through stable training and capturing extensive musical patterns.

Conducted a human listener study where 30% identified AI-generated pieces, and 90% found the music pleasing.

Link to video - https://drive.google.com/file/d/18ImPvYC1iZCmFUCX1c3LaQ44FLRElF6G/view

## Model Architecture

### Generator Network (GenConvNet)
- **Input**: Random noise vector.
- **Layers**:
  - Six transposed convolutional layers.
  - Batch normalization and PReLU activation.
  - Self-attention mechanism after the fourth transposed convolutional layer.
  - Final layer uses a sigmoid activation function.
- **Output**: Generated music sample with dimensions matching the input data.

### Discriminator Network (DiscConvNet)
- **Input**: Music sample (either real or generated).
- **Layers**:
  - Five convolutional layers with PReLU activation and batch normalization.
  - Self-attention mechanism after the third convolutional layer.
  - Dropout layers for regularization.
  - Final linear layer for classification.
- **Output**: Scalar representing the authenticity of the input sample.

### Self-Attention Module
- Implemented as a separate `SelfAttention` class.
- Utilizes query, key, and value convolutions.
- Applies softmax for attention and a learnable parameter gamma for scaling.
- Enhances the model's ability to focus on different parts of the input sequence.
- Improves the coherence and quality of the generated music.

### WGAN-GP
- Utilizes Wasserstein distance for a more stable training of GANs.
- Gradient penalty term added for enforcing the Lipschitz constraint.

### Training Details
- Wasserstein loss with gradient penalty for stable training.
- Separate optimizers for generator and discriminator with Adam optimizer.
- Step learning rate scheduler for both networks.

## Dataset
- Lakh Pianoroll Dataset: A diverse collection of MIDI files, ideal for training music generation models.
- Dataset details: [Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
