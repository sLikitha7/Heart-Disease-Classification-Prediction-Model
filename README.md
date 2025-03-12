# 🏥 Heart Disease Classification Prediction Model

## 📌 Project Overview
This project aims to classify **heart disease risk** based on multiple health indicators using machine learning models. The analysis was conducted in **Google Colab**, leveraging the **dataset_heart_csv** dataset. The goal is to provide insights into **key risk factors** and build an **accurate predictive model** to assist in early diagnosis.

## 🗂️ Dataset Information
- **File Name:** `dataset_heart_csv`
- **Features Include:**
  - **Demographic Data:** Age, Sex
  - **Medical Indicators:** Blood pressure, Cholesterol, Resting ECG results, Max heart rate, ST segment, Thalassemia
  - **Lifestyle & Symptoms:** Chest pain type, Exercise-induced angina, Oldpeak (ST depression)
  - **Target Variable:** `heart_disease` (1 = No Disease,2 = Disease)

## 🚀 Project Workflow
### **1️⃣ Data Preprocessing**
- Loaded dataset using `pandas`
- Checked for missing values and handled them accordingly
- Encoded categorical variables
- Standardized numerical features

### **2️⃣ Exploratory Data Analysis (EDA)**
- **Correlation Heatmap:** Visualized feature relationships
- **Distribution Plots:** Analyzed the distribution of individual features
- **Feature Importance Analysis:** Determined key predictors

### **3️⃣ Model Selection & Training**
- **Algorithms Used:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
  - Gradient Boosting (XGBoost)
- **Model Training:**
  - Split dataset into training and testing sets (80-20 split)
  - Used cross-validation to improve performance
  - Tuned hyperparameters using `GridSearchCV`

### **4️⃣ Model Evaluation**
- **Metrics Used:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
- **Performance Comparison:** Evaluated different models and selected the best-performing one
- **Final Feature Importance Plot:** Identified the most significant predictors

## 📊 Key Findings
- **Strongest Predictors:** 
  - **Chest Pain Type**
  - **Thalassemia**
  - **Max Heart Rate**
  - **Major Vessels**
- **Best Performing Model:** Logistic Regression Performs Best: With 90.7% accuracy and 93.9% recall, it effectively classifies heart disease cases while minimizing false negatives.
- **Accuracy Achieved:** 90.7% 

## 💻 How to Run the Project
### **1️⃣ Setup Google Colab Environment**
- Upload `dataset_heart_csv` to Colab
- Install necessary libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost

## 2️⃣ Run the Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `dataset_heart_csv` file.
3. Load the dataset using the following Python code:

    ```python
    import pandas as pd
    df = pd.read_csv("dataset_heart_csv")
    df.head()
    ```

4. Execute all cells in the Google Collab  Notebook sequentially. The code will:
   - Preprocess data.
   - Train machine learning models.
   - Generate evaluation metrics.

## 3️⃣ Model Interpretation

- **Feature Correlation Heatmap**: Understand feature relationships in the dataset.
- **Feature Importance Plot**: Identify the most influential factors for predicting heart disease.
- **Model Performance Metrics**: Compare models using metrics like accuracy, precision, recall, and ROC-AUC.

## 4️⃣ Results Visualization

- **Distribution Plots**: Analyze how different features impact heart disease predictions.
- **Model Performance Comparison**: Use bar charts to compare models based on accuracy and recall.
- **Model Interpretation**: Identify the best-performing classification approach based on these metrics.

## 🛠️ Future Improvements

- **Optimize Hyperparameters**: Implement Bayesian Optimization for better performance.
- **Deploy as a Web App**: Build a Flask/FastAPI API for real-time disease prediction.
- **Explainable AI (XAI)**: Use SHAP values to interpret model decisions.
- **Automated Data Processing**: Improve preprocessing with AutoML techniques.
- **Cloud Integration**: Deploy the model on Google Cloud or AWS for scalability.
