#  Animal Image Classification Using CNN

## Project Overview

This project is an **Animal Image Classification system using a Convolutional Neural Network (CNN)**. The main problem it solves is automatically identifying an animal from an image instead of identifying it manually. The system takes an uploaded animal image, processes it using a trained CNN model, and predicts the corresponding animal class along with a confidence score.

## Problem Statement

Identifying and organizing a large number of animal images manually can be time-consuming. This project aims to develop an image classification system that can automatically recognize animals from images using deep learning.

##  Solution

A **CNN model** is trained to learn visual features such as edges, shapes, textures, and patterns from animal images. After training, the model can classify an unseen image into one of the animal classes.

##  Dataset

* Approximately **1,900 images**
* **15 animal classes**
* Single-label image classification
* Dataset is divided into:

  * **70% Training**
  * **15% Validation**
  * **15% Testing**

Example classes include:

* Bear
* Bird
* Cat
* Cow
* Other animal classes
* Zebra

## Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **VS Code**
* **Streamlit**

### Python Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* Pillow
* Scikit-learn
* tqdm

##  Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Data Augmentation
   ↓
Train / Validation / Test Split
   ↓
CNN Model
   ↓
Adam Optimizer
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Streamlit Web Application
   ↓
Image Upload
   ↓
Animal Prediction + Confidence
```

##  CNN Model

A **Convolutional Neural Network (CNN)** is used for image classification. The CNN automatically learns important visual features from the images and uses them to classify the animal.

The model is trained using the **Adam optimizer**.

##  Training

During training:

1. Images are given to the CNN.
2. The model generates predictions.
3. The predictions are compared with the actual labels.
4. Loss is calculated.
5. The optimizer updates the model weights.
6. This process is repeated for multiple epochs.

The trained model is saved so that it can be reused without training from the beginning.

Example:

```python
torch.save(model.state_dict(), "cnn_model_epoch10.pth")
```

##  Model Evaluation

The model can be evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

These metrics help determine how well the CNN performs on unseen test images.

##  Streamlit Application

**Streamlit** is used to create the user interface for the trained CNN model.

The user can:

1. Upload an animal image.
2. The image is preprocessed.
3. The trained CNN makes a prediction.
4. The application displays the predicted animal.
5. The confidence score is displayed.

Example:

```text
Predicted Animal: Bear
Confidence: 93.30%
```

### What is Confidence?

Confidence represents how strongly the model favors its predicted class based on the model's output probabilities.

A high confidence does **not always guarantee that the prediction is correct**.

##  Project Structure

```text
cnn_project/
│
├── Dataset/
│   ├── Bear/
│   ├── Bird/
│   ├── Cat/
│   ├── Cow/
│   └── ...
│
├── cnn_env/
│
├── models/
│
├── results/
│
├── train.py
├── eda.py
├── evaluate.py
├── app.py
└── README.md
```

##  How to Run

### 1. Activate the environment

```bash
cnn_env\Scripts\activate
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow scikit-learn tqdm torch torchvision streamlit
```

### 3. Train the model

```bash
python train.py
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in a web browser.

##  Expected Outcome

The final system should be able to:

* Classify animal images automatically.
* Predict one of the 15 animal classes.
* Display the predicted class.
* Display the model's confidence.
* Provide evaluation results using standard classification metrics.

##  Applications

This type of animal classification system can potentially be useful for:

* Wildlife monitoring
* Biodiversity studies
* Organizing animal image datasets
* Educational applications
* Image-based animal identification

##  Future Improvements

Possible improvements include:

* Increasing the dataset size
* Using transfer learning with pretrained models
* Improving data augmentation
* Training for more epochs
* Using a GPU for faster training
* Adding more animal classes
* Improving the Streamlit interface

##  Conclusion

This project demonstrates how **CNN-based deep learning can be used for automatic animal image classification**. The combination of **PyTorch for model development and Streamlit for the user interface** provides a complete workflow from image processing and model training to real-time image prediction.
