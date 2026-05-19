# Predicting Depression from Mental Health Survey Data using Deep Learning

## Project Overview

This project focuses on predicting depression using mental health survey data through Deep Learning techniques. The solution covers the complete machine learning pipeline, including:

• Data preprocessing
• Exploratory Data Analysis (EDA)
• Feature engineering
• Deep learning model building and evaluation
• Hyperparameter tuning
• Bias evaluation
• Deployment using a Streamlit web application

The system helps identify individuals who may be at risk of depression based on demographic, academic, professional, and lifestyle-related factors.


# Domain

Healthcare Analytics | Mental Health Analysis | Deep Learning | Machine Learning


# Problem Statement

Mental health disorders such as depression have become increasingly common among students and working professionals. Early detection of depression is difficult because mental health conditions depend on multiple behavioral, academic, social, and personal factors.

# Business Use Cases

• Depression Risk Prediction
• Mental Health Monitoring
• Early Intervention Support
• Healthcare Decision Support Systems
• Student Wellness Analysis
• Employee Mental Health Assessment

# Dataset Description

## Dataset Source

Mental Health Survey Dataset

## Dataset Size

• Training Data: 140,700 rows × 20 columns
• Testing Data: 93,800 rows × 19 columns

## Key Features

| Column Name                      | Description                  |
| -------------------------------- | ---------------------------- |
| Gender                           | Gender of the individual     |
| Age                              | Age of the individual        |
| Academic Pressure                | Academic stress level        |
| Work Pressure                    | Work stress level            |
| CGPA                             | Academic performance         |
| Study Satisfaction               | Satisfaction with studies    |
| Job Satisfaction                 | Satisfaction with work       |
| Sleep Duration                   | Daily sleep duration         |
| Dietary Habits                   | Food habits                  |
| Degree                           | Educational qualification    |
| Suicidal Thoughts                | History of suicidal thoughts |
| Work/Study Hours                 | Daily working or study hours |
| Financial Stress                 | Financial stress level       |
| Family History of Mental Illness | Family mental health history |
| Depression                       | Target variable              |

---
# Approach

## 1️⃣ Data Preprocessing

• Handling missing values using median and mode imputation
• Removing unnecessary columns
• Encoding categorical variables using Label Encoding
• Feature scaling using StandardScaler
• Train-test splitting for model evaluation

---

## 2️⃣ Exploratory Data Analysis (EDA)

• Univariate analysis
• Distribution analysis
• Correlation analysis
• Class distribution analysis
• Feature relationship visualization

---

## 3️⃣ Feature Engineering

Created meaningful processed features through:

• Numerical feature scaling
• Categorical encoding
• Data transformation pipelines
• Structured input preparation for deep learning

---
## 4️⃣ Deep Learning Model

### 🔹 Deep Learning Architecture

• Input Layer → 17 Features
• Hidden Layer 1 → 128 Neurons + ReLU + Dropout
• Hidden Layer 2 → 64 Neurons + ReLU + Dropout
• Output Layer → Binary Classification

### 🔹 Techniques Used

• PyTorch Deep Learning Framework
• BCEWithLogitsLoss
• Adam Optimizer
• Dropout Regularization

---

## 5️⃣ Hyperparameter Tuning

Tested multiple combinations of:

• Hidden layer sizes
• Dropout values
• Learning rates
• Epoch values

### ✅ Best Configuration

• Hidden Layers → [128, 64]
• Dropout → 0.3
• Learning Rate → 0.001
• Epochs → 100

---

# Model Evaluation

## Evaluation Metrics

• Accuracy
• Precision
• Recall
• F1-score
• Confusion Matrix

# Bias Evaluation

The model fairness was analyzed using:

## Gender-wise Evaluation

• Male prediction accuracy
• Female prediction accuracy

## Age Group Evaluation

• Young
• Adult
• Senior

This helps evaluate whether the model performs fairly across demographic groups.

---

# Model Comparison

The following machine learning models were compared:

• Logistic Regression
• Decision Tree
• Random Forest
• Gradient Boosting
• Support Vector Machine (SVM)
• Deep Learning Model

### Final Observation

Deep Learning achieved highly balanced performance with strong recall and F1-score, making it suitable for mental health prediction tasks.

---

# Model Saving & Pipelines

• Model saved using PyTorch (.pth)
• Preprocessing objects saved using joblib
• Reusable inference pipeline created
• Supports deployment and real-time prediction

Saved artifacts include:

• scaler.pkl
• label_encoders.pkl
• num_imputer.pkl
• cat_imputer.pkl
• feature_columns.pkl
• depression_model.pth

---

# 🌐 Streamlit Application

## 🔹 Features

• User-friendly input interface
• Real-time depression prediction
• Interactive web application

## 🔹 Inputs

• Age
• Gender
• Academic Pressure
• Work Pressure
• Sleep Duration
• Dietary Habits
• Financial Stress
• Family History

## 🔹 Outputs

• Depression Probability
• High Risk / Low Risk Prediction

## 🔹 Tech Stack

• Streamlit
• PyTorch
• Scikit-learn
• Pandas
• NumPy

---

# 📈 Skills Learned

• Data Cleaning & Preprocessing
• Exploratory Data Analysis (EDA)
• Feature Engineering
• Deep Learning using PyTorch
• Hyperparameter Tuning
• Model Evaluation
• Bias Evaluation
• End-to-End ML Pipeline Development
• Streamlit Web App Development
• Model Deployment

---

# 🧰 Technology Stack

| Category         | Tools               |
| ---------------- | ------------------- |
| Programming      | Python              |
| Data Handling    | Pandas, NumPy       |
| Visualization    | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn        |
| Deep Learning    | PyTorch             |
| Model Saving     | Joblib              |
| Web App          | Streamlit           |

---

# 📦 Project Deliverables

• Data preprocessing notebook
• Deep learning model notebook
• Hyperparameter tuning notebook
• Model comparison notebook
• Trained model files (.pth)
• Saved preprocessing artifacts (.pkl)
• Streamlit application (app.py)
• Model evaluation results
• requirements.txt
• Project documentation (README.md)

---

