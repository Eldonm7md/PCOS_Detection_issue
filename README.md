# PCOS Ultrasound Classification Using Convolutional Neural Networks (CNN)

## DA 450 — Deep Learning Final Project

This project implements a complete deep learning pipeline for classifying ovarian ultrasound images into **infected (PCOS)** and **noninfected (normal)** classes using a Convolutional Neural Network (CNN) built from scratch with **TensorFlow/Keras**.

The project follows an end-to-end workflow: dataset understanding, preprocessing, augmentation, imbalance handling, CNN training, evaluation, threshold analysis, and a simple Streamlit prediction interface.

---

## 1. Project Objective

The objective of this project is to develop a CNN-based medical image classification system that can assist in identifying PCOS cases from ultrasound images.

The model predicts whether an input ultrasound image belongs to one of two classes:

- **infected**: ultrasound image showing PCOS characteristics
- **noninfected**: normal ultrasound image

CNN was selected because it is suitable for spatial image data and medical image analysis.

---

## 2. Dataset

### Dataset Source

PCOS Dataset — Figshare  
https://figshare.com/articles/dataset/PCOS_Dataset/27682557

### Dataset Classes

| Class | Number of Images |
|---|---:|
| infected | 6,784 |
| noninfected | 5,000 |
| **Total** | **11,784** |

The dataset shows moderate class imbalance, with more infected images than noninfected images.

---

## 3. Project Workflow

The complete workflow is:

1. Dataset understanding and visualization
2. Data quality checks
3. Image path validation
4. Image resizing
5. Pixel normalization
6. Train/validation/test split
7. Training-only augmentation
8. Class imbalance handling using class weights
9. CNN model training
10. Model evaluation
11. Threshold analysis
12. Streamlit application deployment

---

## 4. Project Structure

```text
DL project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── PCOS/
│   │   └── PCOS/
│   │       ├── infected/
│   │       └── noninfected/
│   │
│   └── processed/
│       ├── clean_pcos_dataframe.csv
│       ├── train_df.csv
│       ├── val_df.csv
│       └── test_df.csv
│
├── figures/
│   ├── class_distribution_bar.png
│   ├── class_distribution_pie.png
│   ├── image_size_scatter.png
│   ├── image_size_hist.png
│   ├── split_class_distribution.png
│   ├── augmented_training_samples.png
│   ├── training_validation_loss.png
│   ├── training_validation_accuracy.png
│   ├── training_validation_precision.png
│   ├── training_validation_recall.png
│   ├── confusion_matrix_threshold_050.png
│   ├── confusion_matrix_best_threshold.png
│   ├── roc_curve.png
│   ├── threshold_analysis.png
│   ├── cnn_architecture.png
│   └── workflow_diagram.png
│
├── models/
│   ├── best_pcos_cnn_model.keras
│   └── final_pcos_cnn_model.keras
│
├── PCOS_notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
├── README.md
└── final_report.pdf
```

---

## 5. Notebooks Execution Order

Run the notebooks in the following order:

### 1. `data_understanding.ipynb`

This notebook performs:

- dataset loading
- dataframe creation
- class distribution analysis
- missing value checks
- duplicate checks
- corrupted image checks
- image dimension analysis
- sample image visualization
- saving clean dataset metadata

### 2. `preprocessing.ipynb`

This notebook performs:

- label encoding
- stratified train/validation/test split
- image resizing
- normalization
- training-only augmentation
- class weight calculation
- saving train, validation, and test CSV files

### 3. `model_training.ipynb`

This notebook performs:

- CNN model construction
- model compilation
- class-weighted training
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- training curve visualization
- model saving

### 4. `evaluation.ipynb`

This notebook performs:

- final test set evaluation
- classification report
- confusion matrix
- ROC curve
- AUC calculation
- threshold analysis
- final prediction saving

---

## 6. Preprocessing Details

The preprocessing pipeline includes:

### Data Cleaning

- Checked missing values
- Checked duplicate rows
- Checked corrupted/unreadable images
- Validated image file paths

### Image Resizing

All images were resized to:

```text
224 × 224 × 3
```

This creates a fixed input shape for the CNN.

### Normalization

Pixel values were normalized from `[0, 255]` to `[0, 1]` using:

```text
x = x / 255
```

### Train/Validation/Test Split

The dataset was split as follows:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The split was stratified to preserve class proportions.

### Data Augmentation

Augmentation was applied only to the training set to avoid data leakage.

Augmentation techniques:

- rotation
- width shift
- height shift
- zoom
- horizontal flip

### Imbalance Handling

The dataset showed moderate class imbalance.  
Class weights were computed from the training set and used during model training.

Class weights used:

```python
{0: 1.1782857142857144, 1: 0.8685762426284751}
```

Where:

- `0 = noninfected`
- `1 = infected`

---

## 7. CNN Model Architecture

The CNN model was implemented from scratch using TensorFlow/Keras.  
No pre-trained weights were used.

### Architecture Summary

| Stage | Description |
|---|---|
| Input | 224 × 224 × 3 RGB image |
| Conv Block 1 | Conv2D 32 filters, BatchNormalization, MaxPooling |
| Conv Block 2 | Conv2D 64 filters, BatchNormalization, MaxPooling |
| Conv Block 3 | Conv2D 128 filters, BatchNormalization, MaxPooling |
| Conv Block 4 | Conv2D 256 filters, BatchNormalization, MaxPooling |
| Feature Reduction | GlobalAveragePooling2D |
| Dense Layer | Dense 128, ReLU, L2 regularization |
| Regularization | Dropout 0.4 |
| Output | Dense 1, Sigmoid activation |

The sigmoid output produces a probability between 0 and 1.

---

## 8. Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 25 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Dropout Rate | 0.4 |
| Regularization | L2 |
| EarlyStopping Patience | 5 |
| ReduceLROnPlateau Factor | 0.3 |

---

## 9. Loss Function

Because the task is binary classification, the model uses **Binary Crossentropy**.

```text
L = -(1/N) Σ [ y log(y_hat) + (1-y) log(1-y_hat) ]
```

Where:

- `y` is the true label
- `y_hat` is the predicted probability
- `N` is the number of samples

---

## 10. Model Training

Two CNN experiments were conducted.

### Experiment 1 — Final Selected Model

The first CNN model used:

- 224 × 224 input size
- four convolutional blocks
- dropout 0.4
- L2 regularization
- class weights

This model achieved the most stable validation performance and was selected as the final model.

### Experiment 2 — Improved Generalization Trial

A second experiment was tested using:

- smaller input size
- stronger augmentation
- simpler architecture
- higher dropout

Although it trained faster, it showed lower validation stability.  
Therefore, Experiment 1 was selected as the final model.

---

## 11. Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Confusion Matrix
- Threshold Analysis

Accuracy alone was not considered sufficient because this is a medical image classification task with class imbalance.

---

## 12. Final Evaluation Results

At the selected threshold of `0.10`, the model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 89.8% |
| Precision | 100% |
| Recall | 82.2% |
| F1-score | 90.2% |

### Confusion Matrix at Threshold 0.10

|  | Predicted noninfected | Predicted infected |
|---|---:|---:|
| True noninfected | 750 | 0 |
| True infected | 181 | 837 |

The model produced no false positives at this threshold, but some infected cases were missed.  
This was discussed as an important limitation in the medical interpretation.

---

## 13. Threshold Analysis

The default binary classification threshold is `0.50`.

However, in medical classification, recall is important because false negatives may represent missed PCOS cases.

Therefore, threshold analysis was performed from `0.10` to `0.90`.

The best threshold based on F1-score was:

```text
0.10
```

This threshold improved recall and achieved the highest F1-score.

---

## 14. Streamlit Application

A simple Streamlit application was developed to demonstrate model deployment.

The application allows the user to:

1. Upload an ultrasound image
2. Preprocess the image
3. Run prediction using the trained CNN model
4. Display the predicted class
5. Display the prediction probability

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 15. Installation

### Step 1: Create or activate Python environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Step 2: Install required libraries

```bash
pip install -r requirements.txt
```

---

## 16. Requirements

Main libraries:

- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- opencv-python
- pillow
- streamlit
- jupyter

See `requirements.txt` for the complete dependency list.

---

## 17. How to Reproduce Results

1. Download the dataset from Figshare.
2. Place the dataset inside:

```text
data/PCOS/PCOS/
```

3. Ensure the folder structure is:

```text
data/PCOS/PCOS/infected/
data/PCOS/PCOS/noninfected/
```

4. Run notebooks in order:

```text
1. data_understanding.ipynb
2. preprocessing.ipynb
3. model_training.ipynb
4. evaluation.ipynb
```

5. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

---

## 18. Important Notes

- The model was implemented using TensorFlow/Keras only.
- No pre-trained weights were used.
- Test data was used only during final evaluation.
- Augmentation was applied only to training data.
- Class weights were used to handle imbalance.
- Threshold analysis was used to improve medical interpretation.

---

## 19. Limitations

- The model is not intended for real clinical diagnosis without medical validation.
- The dataset may not represent all ultrasound machines or patient populations.
- Some infected cases were still missed, as shown by false negatives.
- Further clinical testing is required before real-world use.

---

## 20. Future Work

Possible future improvements:

- use larger and more diverse medical datasets
- apply Grad-CAM for explainability
- compare with transfer learning models
- improve recall further
- deploy the system as a web or mobile application
- validate results with medical experts

---

## 21. Team Members

| Name | Role |
|---|---|
| Emam & Mohammad | Data Understanding and Visualization |
| Emam & Mohammad | Preprocessing and Data Cleaning |
| Mohammad | | Streamlit App |
| Student 4 | Presentation & Video |
| Student 5 | Report |

---

## 22. References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. Keras Documentation: https://keras.io/
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Streamlit Documentation: https://streamlit.io/
5. PCOS Dataset — Figshare: https://figshare.com/articles/dataset/PCOS_Dataset/27682557
