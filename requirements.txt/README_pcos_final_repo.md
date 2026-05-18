# PCOS Ultrasound Classification Using Convolutional Neural Networks (CNN)

## DA 450 — Deep Learning Final Project

This project implements an end-to-end deep learning pipeline for classifying ovarian ultrasound images into **infected (PCOS)** and **noninfected (normal)** classes using a Convolutional Neural Network (CNN) built from scratch with **TensorFlow/Keras**.

The project includes dataset understanding, preprocessing, augmentation, imbalance handling, CNN training, experimental model comparison, final evaluation, threshold analysis, and a Streamlit prediction interface.

> **Important note:** This project is for academic and educational purposes only. It is not intended to replace clinical diagnosis or medical expert evaluation.

---

## 1. Project Objective

The objective of this project is to build a CNN-based medical image classification model that can assist in detecting PCOS from ovarian ultrasound images.

The classification task is binary:

| Class | Meaning |
|---|---|
| **infected** | Ultrasound image showing PCOS characteristics |
| **noninfected** | Normal ultrasound image |

CNNs were selected because they are well suited for spatial image data and can learn visual patterns such as edges, textures, shapes, and deeper image features.

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

The dataset contains a moderate class imbalance, with more infected images than noninfected images.  
Class weights were used during training to reduce bias toward the majority class.

---

## 3. Repository Structure

The final repository structure is organized as follows:

```text
DL project/
│
├── app/
│   ├── __pycache__/
│   └── streamlit_app.py
│
├── data/
│   └── PCOS/
│       └── PCOS/
│           ├── infected/
│           └── noninfected/
│
├── processed/
│   ├── clean_pcos_dataframe.csv
│   ├── train_df.csv
│   ├── val_df.csv
│   └── test_df.csv
│
├── figures/
│   ├── augmented_training_samples.png
│   ├── class_distribution_bar.png
│   ├── class_distribution_pie.png
│   ├── confusion_matrix_best_threshold.png
│   ├── confusion_matrix_threshold_050.png
│   ├── final_model_comparison_table.png
│   ├── image_size_hist.png
│   ├── image_size_scatter.png
│   ├── metrics_threshold_0.5.csv
│   ├── roc_curve_final_v3.png
│   ├── roc_curve.png
│   ├── samples_infected.png
│   ├── samples_noninfected.png
│   ├── split_class_distribution.png
│   ├── test_metrics_threshold_050.csv
│   ├── test_predictions.csv
│   ├── threshold_analysis_results.csv
│   ├── threshold_analysis.png
│   ├── training_validation_accuracy.png
│   ├── training_validation_loss.png
│   ├── training_validation_precision.png
│   ├── training_validation_recall.png
│   ├── V1_model_3dARCH.png
│   ├── V3_MODEL_3dARCH.png
│   ├── V3_Regularized_CNN_accuracy.png
│   ├── V3_Regularized_CNN_auc.png
│   ├── V3_Regularized_CNN_loss.png
│   ├── V3_Regularized_CNN_precision.png
│   └── V3_Regularized_CNN_recall.png
│
├── models/
│   ├── best_pcos_cnn_model.keras
│   ├── best_pcos_cnn_model_v2.keras
│   ├── final_pcos_cnn_model.keras
│   ├── v3_regularized_cnn_final.keras
│   └── v3_regularized_cnn_finalkeras
│
├── PCOS_notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
├── final_report/
├── README.md
├── README_PCOS_CNN.md
├── .gitignore
└── PCOS.zip
```

---

## 4. Project Workflow

The project follows this workflow:

1. Dataset understanding and visualization
2. Data quality checks
3. Image path validation
4. Image resizing
5. Pixel normalization
6. Stratified train/validation/test split
7. Training-only augmentation
8. Class imbalance handling using class weights
9. CNN model training
10. Experimental comparison between CNN versions
11. Final model selection
12. Test set evaluation
13. ROC curve and confusion matrix analysis
14. Threshold analysis for medical interpretation
15. Streamlit application deployment

---

## 5. Notebooks Execution Order

Run the notebooks in the following order:

### 1. `data_understanding.ipynb`

This notebook performs:

- dataset loading
- dataframe creation
- class distribution analysis
- image dimension analysis
- sample image visualization
- missing value checks
- duplicate checks
- corrupted image checks
- saving cleaned metadata

### 2. `preprocessing.ipynb`

This notebook performs:

- image path validation
- stratified train/validation/test split
- preprocessing preparation
- class weight calculation
- saving train, validation, and test CSV files into `processed/`

### 3. `model_training.ipynb`

This notebook performs:

- CNN model construction
- model compilation
- class-weighted training
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- training curve visualization
- comparison between V1, V2, and V3
- saving the final selected model

### 4. `evaluation.ipynb`

This notebook performs:

- loading the final selected V3 model
- final test set evaluation
- correcting label interpretation for medical evaluation
- classification report
- confusion matrix
- ROC curve
- AUC calculation
- threshold analysis
- final threshold selection

---

## 6. Preprocessing Details

### Data Cleaning

The following checks were performed:

- missing values
- duplicate records
- corrupted images
- invalid image paths
- class distribution
- image size distribution

### Image Size

The final V3 model uses:

```text
224 × 224 × 3
```

### Normalization

The final V3 model includes normalization inside the model using:

```python
layers.Rescaling(1./255)
```

Therefore, during final evaluation, the test image generator should **not** apply another `rescale=1./255`, because double normalization causes incorrect predictions.

### Train/Validation/Test Split

The dataset was split as follows:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The split was stratified to preserve class proportions.

### Data Augmentation

Augmentation was applied only to the training stage to avoid data leakage.

The final V3 model used ultrasound-oriented augmentation:

- horizontal flip
- rotation
- translation
- zoom
- contrast variation
- brightness variation

These transformations were selected because ultrasound images may vary due to acquisition angle, device settings, position, depth, and lighting/gain differences.

---

## 7. Class Label Handling

During model training, the generator used this class order:

```text
infected = 0
noninfected = 1
```

In Keras binary classification, class index `1` is treated as the positive class for built-in metrics such as precision and recall.  
Therefore, raw `model.evaluate()` precision and recall describe the `noninfected` class, not the infected class.

For medical interpretation, the evaluation notebook converts the labels so that:

```text
noninfected = 0
infected = 1
```

This makes precision, recall, F1-score, confusion matrix, ROC curve, and threshold analysis correctly represent PCOS detection.

---

## 8. Final CNN Model Architecture

The final selected model is:

```text
V3 Regularized CNN
```

Model file:

```text
models/v3_regularized_cnn_final.keras
```

The model was built from scratch using TensorFlow/Keras.  
No pretrained weights were used.

### Architecture Summary

| Stage | Description |
|---|---|
| Input | 224 × 224 × 3 ultrasound image |
| Preprocessing | Rescaling + medical augmentation |
| CNN Block 1 | Conv2D 32, BatchNormalization, ReLU, MaxPooling |
| CNN Block 2 | Conv2D 64, BatchNormalization, ReLU, MaxPooling |
| CNN Block 3 | Conv2D 128, BatchNormalization, ReLU, MaxPooling, Dropout |
| CNN Block 4 | Conv2D 256, BatchNormalization, ReLU, MaxPooling, Dropout |
| Feature Reduction | GlobalAveragePooling2D |
| Dense Head | Dense 128, ReLU, L2 Regularization, Dropout |
| Output | Dense 1, Sigmoid |

---

## 9. Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 25 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Regularization | L2 |
| Dropout | Used in deeper CNN blocks and dense layer |
| EarlyStopping | Used |
| ReduceLROnPlateau | Used |
| ModelCheckpoint | Used |

---

## 10. Model Experiments

Three CNN experiments were conducted.

### Experiment 1 — Baseline CNN (V1)

V1 was the first stable baseline CNN.

| Metric | Result |
|---|---:|
| Validation Accuracy | 0.8835 |
| Validation AUC | 0.9348 |
| Validation Recall | 0.7976 |
| Validation Loss | 0.7914 |

V1 achieved strong baseline performance and stable training behavior.

### Experiment 2 — Smaller CNN (V2)

V2 used a smaller input size and simpler architecture.

| Metric | Result |
|---|---:|
| Validation Accuracy | 0.8179 |
| Validation AUC | 0.9454 |
| Validation Recall | 0.7849 |
| Validation Loss | 0.4352 |

Although V2 trained faster and achieved good early AUC, its validation behavior became unstable.  
Therefore, it was not selected as the final model.

### Experiment 3 — Regularized CNN (V3)

V3 improved the baseline CNN using stronger regularization and more realistic ultrasound augmentation.

| Metric | Result |
|---|---:|
| Validation Accuracy | 0.9089 |
| Validation AUC | 0.9832 |
| Validation Loss | 0.2441 |

V3 achieved the best validation performance and was selected as the final model.

---

## 11. Final Model Comparison

| Model | Accuracy | AUC | Recall | Loss | Main Observation | Decision |
|---|---:|---:|---:|---:|---|---|
| V1 Baseline CNN | 0.8835 | 0.9348 | 0.7976 | 0.7914 | Strong stable baseline | Baseline reference |
| V2 Smaller CNN | 0.8179 | 0.9454 | 0.7849 | 0.4352 | Fast but unstable | Not selected |
| V3 Regularized CNN | 0.9089 | 0.9832 | — | 0.2441 | Best overall validation performance | Final selected model |

> The V3 validation precision and recall from `model.evaluate()` were not used directly for final medical interpretation because of the binary class ordering issue explained above.

---

## 12. Final Test Evaluation

After medical label correction, the final model was evaluated with:

```text
0 = noninfected
1 = infected
```

At the default threshold `0.50`, the model achieved strong performance.  
Threshold analysis was then used to select a better operating threshold for medical decision-making.

---

## 13. Threshold Analysis

The default binary threshold is `0.50`.

However, in medical image classification, threshold selection is important because false negatives represent missed infected cases.

Thresholds from `0.10` to `0.90` were tested using:

- precision
- recall
- F1-score

The final selected threshold was:

```text
0.40
```

Threshold `0.40` was selected because it provided the best F1-score and a strong balance between precision and recall.

---

## 14. Final Confusion Matrix at Threshold 0.40

Using the corrected medical interpretation:

```text
0 = noninfected
1 = infected
```

Final confusion matrix:

|  | Predicted noninfected | Predicted infected |
|---|---:|---:|
| True noninfected | 738 | 12 |
| True infected | 94 | 924 |

Interpretation:

- 738 noninfected images were correctly classified.
- 924 infected images were correctly classified.
- 12 noninfected images were incorrectly classified as infected.
- 94 infected images were incorrectly classified as noninfected.

This threshold achieved a strong balance between reducing missed PCOS cases and maintaining high precision.

---

## 15. Final Medical Metrics at Threshold 0.40

| Metric | Score |
|---|---:|
| Accuracy | 0.9400 |
| Precision | 0.9872 |
| Recall | 0.9077 |
| F1-score | 0.9458 |
| AUC | 0.9780 |

These results show that the final V3 model has strong classification performance and good medical screening capability within the scope of this academic project.

---

## 16. Streamlit Application

A simple Streamlit app was created to demonstrate prediction deployment.

The app allows the user to:

1. Upload an ultrasound image
2. Preprocess the image
3. Run prediction using the trained CNN model
4. Display the predicted class
5. Display the prediction probability
6. Show a medical disclaimer

Run the app:

```bash
streamlit run app/streamlit_app.py
```

---

## 17. Installation

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

## 18. How to Reproduce Results

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

## 19. Requirements

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

## 20. Important Notes

- The model was implemented using TensorFlow/Keras only.
- No pretrained weights were used.
- Test data was used only during final evaluation.
- Augmentation was applied only to training data.
- Class weights were used to handle imbalance.
- Threshold analysis was used to improve medical interpretation.
- Label correction was required for final medical metrics because training used `infected = 0` and `noninfected = 1`.
- The Streamlit app is a demonstration interface and not a clinical tool.

---

## 21. Limitations

- The model is not intended for real clinical diagnosis.
- The dataset may not represent all ultrasound machines, hospitals, or patient populations.
- Some infected cases were still missed.
- External clinical validation is required before any real-world use.
- Explainability methods such as Grad-CAM were not included in the final project scope.

---

## 22. Future Work

Possible future improvements:

- use larger and more diverse medical datasets
- apply Grad-CAM for explainability
- compare with transfer learning models
- improve recall while preserving precision
- validate results with medical experts
- deploy the system as a web or mobile application

---

## 23. Team Members

| Name | Role |
|---|---|
| Mohammed Noor Faris Hawwari | Model development and coding |
| Imam Mohsen Abu Haifa | Model development and coding |
| Abdulrahman Darrar Aedeh | Presentation and demo video |
| Mohammad Hijazi Al-Zoubi | Streamlit interface |
| Mohammad Emad Khmaiseh | Report and documentation |

---

## 24. References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. Keras Documentation: https://keras.io/
3. Scikit-learn Documentation: https://scikit-learn.org/
4. Streamlit Documentation: https://streamlit.io/
5. PCOS Dataset — Figshare: https://figshare.com/articles/dataset/PCOS_Dataset/27682557
