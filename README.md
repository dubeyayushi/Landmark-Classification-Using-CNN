# Landmark Classification Project

This repository contains the code and notebooks for the Landmark Classification project, where we build a Convolutional Neural Network (CNN) to classify landmarks from images. The project involves three major steps: building a CNN from scratch, applying transfer learning, and deploying the model in a simple app.

## Project Overview

Photo sharing and storage services often rely on location metadata to organize and tag images. However, not all images contain this metadata. This project aims to solve this problem by creating a machine learning model that can predict the location of an image based on the landmarks depicted in it.

The project is divided into three main stages:

1. **Create a CNN from Scratch**: 
   - Visualize and preprocess the dataset.
   - Build a CNN from scratch to classify the landmarks.
   - Export the best model using Torch Script.

2. **Create a CNN using Transfer Learning**: 
   - Explore different pre-trained models.
   - Use transfer learning to train and test a network for this classification task.
   - Export the best transfer learning model using Torch Script.

3. **Deploy the Model in an App**: 
   - Use the best model to create a simple app.
   - Test the app for identifying landmarks in images.
   - Generate an archive file for submission.

## Project Structure

The repository contains the following key files and directories:

- **cnn_from_scratch.ipynb**: Notebook for creating a CNN from scratch.
- **transfer_learning.ipynb**: Notebook for applying transfer learning.
- **app.ipynb**: Notebook for deploying the model in a simple app.
- **data/**: Directory containing the dataset (note: dataset files are not included in the repository).
- **models/**: Directory for saving trained models and exported Torch Script models.
- **README.md**: This file.

## Getting Started

### Prerequisites

To run the notebooks, you need the following software installed:

- Python 3.x
- Jupyter Notebook
- PyTorch
- Torchvision
- Other Python dependencies listed in `requirements.txt`

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/landmark-classification.git
cd landmark-classification
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. **CNN from Scratch**: Open `cnn_from_scratch.ipynb` in Jupyter Notebook and run the cells to build and train a CNN from scratch.
    
2. **Transfer Learning**: Open `transfer_learning.ipynb` in Jupyter Notebook and run the cells to explore pre-trained models and apply transfer learning.
    
3. **Deploying the Model**: Open `app.ipynb` in Jupyter Notebook and run the cells to deploy your model in a simple app.
    
## Results

- **CNN from Scratch**: Achieved 62% accuracy on the test set.
- **Transfer Learning**: Achieved 75% accuracy on the test set using ResNet18.

## Deployment

The app created in `app.ipynb` allows users to upload an image and get predictions for the most likely landmarks depicted in the image.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was completed as part of the advanced Machine Learning Nanodegree Program by Udacity and AWS.
