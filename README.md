# Pneumonia Classifier using Deep Learning

## Project Overview

This project builds a deep learning model to classify chest X-ray images as either **Normal** or **Pneumonia**. The aim is to support medical image analysis by using convolutional neural networks and transfer learning to detect pneumonia from X-ray scans.

The project uses the **Chest X-Ray Pneumonia dataset** from Kaggle and compares model performance using evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

Dataset: **Chest X-Ray Images (Pneumonia)**
Source: Kaggle

The dataset is divided into:

* Training set
* Validation set
* Test set

Classes:

* `NORMAL`
* `PNEUMONIA`

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV
* Kaggle / Jupyter Notebook

## Model Approach

The project uses a transfer learning approach with **VGG16** pretrained on ImageNet.

Main steps:

1. Load and preprocess chest X-ray images.
2. Resize images to `224x224`.
3. Convert grayscale images to RGB format.
4. Apply image augmentation to reduce overfitting.
5. Use VGG16 as a feature extractor.
6. Add custom classification layers.
7. Train the model on the training dataset.
8. Validate performance using the validation set.
9. Evaluate final results on the test set.

## Model Architecture

The model uses:

* VGG16 base model without the top layer
* Global Average Pooling layer
* Dense layer with ReLU activation
* Dropout layer
* Final Dense layer with sigmoid activation for binary classification

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

These metrics help assess how well the model identifies pneumonia cases and avoids false predictions.

## Results

The model was trained and tested using the Chest X-ray dataset. Performance was evaluated on the test set to measure how well the model generalises to unseen images.

The final results include:

* Training and validation accuracy/loss curves
* Confusion matrix
* Precision, recall, and F1-score
* Example predictions

## Grad-CAM Visualisation

Grad-CAM was used to visualise which areas of the X-ray image influenced the model’s prediction. This helps improve interpretability by showing the regions the model focused on when classifying an image as normal or pneumonia.

## Project Structure

```text
pneumonia-classifier/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── models/
│   └── best_model.h5
├── results/
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix.png
│   └── gradcam_examples.png
└── data/
    ├── train/
    ├── val/
    └── test/
```

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/pneumonia-classifier.git
```

2. Open the project folder:

```bash
cd pneumonia-classifier
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Run the notebook cells in order.

## Requirements

Example `requirements.txt`:

```text
tensorflow
keras
numpy
matplotlib
scikit-learn
opencv-python
pandas
```

## Limitations

This project is for educational purposes only. The model should not be used as a replacement for professional medical diagnosis. Real-world clinical use would require larger datasets, medical validation, regulatory approval, and expert review.

## Future Improvements

Possible improvements include:

* Testing additional pretrained models such as ResNet50, EfficientNet, or DenseNet
* Applying stronger data augmentation
* Using cross-validation
* Improving Grad-CAM visualisations
* Fine-tuning more layers of the pretrained model
* Deploying the model as a web application

## Conclusion

This project demonstrates how deep learning and transfer learning can be applied to medical image classification. The VGG16-based model provides a practical approach for detecting pneumonia from chest X-ray images and supports interpretability through Grad-CAM visualisation.

