# 🏥 Heart Disease Classification Prediction Model

## 📌 Project Overview
This project aims to classify **heart disease risk** based on multiple health indicators using machine learning models. The analysis was conducted in **Google Colab**, leveraging the **dataset_heart_csv** dataset. The goal is to provide insights into **key risk factors** and build an **accurate predictive model** to assist in early diagnosis.

## 🗂️ Dataset Information
- **File Name:** `dataset_heart_csv`
- **Features Include:**
  - **Demographic Data:** Age, Sex
  - **Medical Indicators:** Blood pressure, Cholesterol, Resting ECG results, Max heart rate, ST segment, Thalassemia
  - **Lifestyle & Symptoms:** Chest pain type, Exercise-induced angina, Oldpeak (ST depression)
  - **Target Variable:** `heart_disease` (1 = Disease, 0 = No Disease)

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
  2️⃣ Run the Notebook
Open Google Colab and upload the dataset_heart_csv file.
Load the dataset using:
python
Copy
Edit
import pandas as pd
df = pd.read_csv("dataset_heart_csv")
df.head()
Execute all the cells in the Jupyter Notebook sequentially.
The code will preprocess data, train machine learning models, and generate evaluation metrics.
3️⃣ Model Interpretation
Check the Feature Correlation Heatmap to understand feature relationships.
View the Feature Importance Plot to determine the most influential factors.
Compare model performance using accuracy, precision, recall, and ROC-AUC.
4️⃣ Results Visualization
Analyze distribution plots to see how different features impact heart disease.
Evaluate model performance using bar charts comparing accuracy and recall.
Interpret model results to identify the best-performing classification approach.
🛠️ Future Improvements
Optimize Hyperparameters: Implement Bayesian Optimization for better performance.
Deploy as a Web App: Build a Flask/FastAPI API for real-time disease prediction.
Explainable AI (XAI): Use SHAP values to interpret model decisions.
Automated Data Processing: Improve preprocessing with AutoML techniques.
Cloud Integration: Deploy the model on Google Cloud or AWS for scalability.
📜 License
This project is licensed under the MIT License.
