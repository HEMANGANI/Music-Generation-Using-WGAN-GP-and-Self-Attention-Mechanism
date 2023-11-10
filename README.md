# Music Generation with GANs

This repository contains the implementation of a music generation model using Generative Adversarial Networks (GANs) specifically designed to create melodic compositions. The model leverages the power of Convolutional LSTMs and is built using PyTorch, focusing on sequential data processing.

## Project Overview

This project aims to explore the capabilities of GANs in the realm of music generation. Utilizing the Lakh Pianoroll Dataset, the model applies advanced techniques such as Wasserstein GANs with Gradient Penalty (WGAN-GP) and Progressive Growing of GANs to generate high-quality and coherent musical sequences.

## Dataset

The Lakh Pianoroll Dataset is used for training the model, which provides a rich collection of MIDI files suitable for training music generation models. More information about the dataset can be found [here](https://salu133445.github.io/lakh-pianoroll-dataset/).

## Model Architecture

The model architecture consists of:
- **Convolutional LSTM layers**: For capturing the temporal dynamics of music.
- **WGAN-GP**: An improved version of GANs that provides stability during training.
- **Progressive GANs**: A training methodology that starts with low-resolution data and progressively moves to higher resolutions.
