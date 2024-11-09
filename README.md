# Face Detection and Recognition Using Haar Cascade Classifier and VGG

This project presents a face detection and recognition system using OpenCV’s Haar Cascade Classifier for face detection, a pre-trained VGG model for feature extraction, and a classification head to create and manage face embeddings. The system organizes embeddings in a database and classifies new faces using a K-Nearest Neighbors (KNN) approach with cosine similarity.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
This project integrates face detection and recognition for applications requiring accurate, efficient identification. Using OpenCV’s Haar Cascade Classifier and a deep learning pipeline with a pre-trained VGG model, it extracts a unique embedding for each face and uses these embeddings in a structured database. Classification relies on a K-Nearest Neighbors (KNN) approach with cosine similarity for robust and reliable face matching.

## Methodology

1. **Face Detection Using Haar Cascade Classifier**:
   - The initial face detection uses OpenCV's Haar Cascade Classifier, identifying and isolating face regions within images.

2. **Feature Extraction with VGG Model**:
   - Each detected face is processed by a pre-trained VGG model to extract a **2622-dimensional vector** (embedding), representing unique features of the face.
   - This high-dimensional vector serves as the initial "signature" for each face.

3. **Classification Head for Embeddings**:
   - A classification head, implemented in PyTorch, reduces each 2622-dimensional vector to a **256-dimensional embedding**. This 256-length vector represents a compressed, discriminative form of the face signature, optimized for classification.
   - These 256-dimensional embeddings form the core of the database, ensuring efficient storage and quick retrieval.

4. **Creating Class Vectors in the Embeddings Database**:
   - Each identity in the database has multiple instances (images) represented by embeddings.
   - To create a single, stable "class vector" for each identity, the system calculates the **average vector of the normalized embeddings** for all instances of that identity. This averaged vector serves as the representative signature for the class in the database.

5. **Face Recognition with KNN and Cosine Similarity**:
   - When a new face is detected and converted to a 256-dimensional embedding, the system compares it against the class vectors in the embeddings database.
   - **K-Nearest Neighbors (KNN) with Cosine Similarity** is used to match the new embedding to the closest class vector, identifying the individual or marking it as unknown if no close match is found.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- PyTorch (for classification head and embedding processing)
- NumPy
- Jupyter Notebook (for running the notebook interactively)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/face-detection-recognition
   ```
   
2. Install the required packages:
   ```bash
   pip install opencv-python numpy torch torchvision
   ```

3. Run the notebook:
   - Open the Jupyter Notebook (`Face_detection_and_recognition.ipynb`) to get started.

## Usage

1. **Prepare Database**: Place images of individuals to be recognized in the -images_path- directory.
2. **Run the Notebook**:
   - The notebook will detect faces, extract embeddings, and populate the embeddings database with class vectors for each identity.
   - Run cells to train the classification head and set up the KNN classification.
   - Test recognition on new image.
   
