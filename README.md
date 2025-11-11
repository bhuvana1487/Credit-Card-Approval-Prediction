# Credit Card Fraud Detection

## 📋 Project Overview

This project implements machine learning models to detect fraudulent credit card transactions in real-time, improving security and reducing financial losses for financial institutions.

## Business Problem
Develop a robust ML system to identify fraudulent transactions while minimizing false positives to ensure customer satisfaction and reduce financial losses.

## Dataset Features
**Transaction Information:** Amount, merchant category, location

**Cardholder Details:** Age, income, spending patterns

**Contextual Data:** Device type, day of week, region

**Target Variable:** Is_Fraudulent (Binary: Fraud/Non-Fraud)

## 📊 Dataset Description

### Features Overview

### Numerical Features:

**Transaction_Amount:** Value of the transaction

**Cardholder_Age:** Age of the cardholder

**Cardholder_Monthly_Income:** Monthly income of cardholder

**Cardholder_Average_Spend:** Average spending pattern

**Credit_Limit:** Card credit limit

### Categorical Features:

**Card_Type:** Type of credit card

**Merchant_Category:** Category of merchant

**Location:** Transaction location

**Region:** Geographical region

**Cardholder_Gender:** Gender of cardholder

**Device_Type:** Device used for transaction

**Day_of_Week:** Day when transaction occurred

**Identifier Columns:**

**Transaction_ID:** Unique transaction identifier

**Transaction_DateTime:** Timestamp of transaction

**Target Variable:**

Is_Fraudulent: 0 = Legitimate, 1 = Fraudulent

## 🔍 Exploratory Data Analysis

Data Quality Assessment

**Missing Values:** Handled using KNN imputation for numerical features and mode imputation for categorical features

**Duplicates:** Identified and removed

**Class Imbalance:** Significant imbalance between fraudulent and legitimate transactions

**Key Insights from EDA**

**Univariate Analysis**

**Fraud Distribution:** Highly imbalanced dataset with rare fraud cases

**Transaction Amount:** Right-skewed distribution, requiring log transformation

**Merchant Categories:** Certain categories show higher fraud rates

**Cardholder Age:** Normal distribution with concentration in 25-55 age range

**Bivariate Analysis**

**Transaction Amount vs Fraud:** Higher amounts show increased fraud probability

**Merchant Category Impact:** Specific categories have significantly higher fraud rates

**Time Patterns:** Certain days show higher fraud incidence

### Multivariate Analysis

Complex interactions between transaction amount, merchant category, and location

Regional patterns in fraud distribution

Device type correlations with fraud probability

## ⚙️ Data Preprocessing

### Handling Missing Values

**Numerical features: KNN Imputation**

**Categorical features: Mode imputation**

## Outlier Treatment

* Identified outliers using IQR method in Transaction_Amount

* Applied capping for extreme values

* Considered business context for outlier handling

**Skewness Correction**

Log transformation for highly skewed features

**Target Variable Processing**

Convert to numeric encoding

## 🧮 Feature Engineering & Encoding

### Categorical Feature Encoding

Identify categorical features

### Apply one-hot encoding

## Feature Selection Methods
**Correlation Analysis**

* Calculated correlation with target variable

* Identified highly correlated features for fraud detection

**Random Forest Feature Importance**

**Selected Top Features**

1. Transaction_Amount

2. Merchant_Category_encoded

3. Cardholder_Average_Spend

4. Credit_Limit

5. Cardholder_Age

6. Location_encoded

7. Device_Type_encoded

8. Cardholder_Monthly_Income

9. Region_encoded

10. Day_of_Week_encoded

## 🤖 Model Building

**Algorithms Implemented**

Logistic Regression

Naive Bayes (Gaussian)

Decision Tree Classifier

Random Forest Classifier

K-Nearest Neighbors

Support Vector Machine

## Preprocessing Pipeline

** Numerical feature scaling**

**Handling Class Imbalance**

Apply SMOTE to address class imbalance

## 📈 Model Evaluation

**Performance Metrics Comparison**
**Model	Accuracy	Precision	Recall	F1-Score**
Random Forest	95.2%	0.91	0.88	0.89
Logistic Regression	92.1%	0.87	0.82	0.84
Decision Tree	93.5%	0.89	0.85	0.87
K-Nearest Neighbors	90.8%	0.84	0.79	0.81
Support Vector Machine	91.7%	0.86	0.81	0.83
Naive Bayes	88.3%	0.81	0.76	0.78

## Key Findings
**Random Forest** achieved the best overall performance with highest F1-score

**High Precision** is crucial to minimize false positives in fraud detection

**Class imbalance handling** significantly improved model recall

**Feature engineering** played critical role in model performance

## Confusion Matrix Analysis

**True Positives:** Correctly identified fraudulent transactions

**False Positives:** Legitimate transactions flagged as fraud (customer impact)

**False Negatives:** Missed fraudulent transactions (financial loss)

**True Negatives:** Correctly identified legitimate transactions

**🎯 Business Impact**

**Achieved Objectives**

**✅ High Detection Rate:** Significant improvement in fraud identification

**✅ Reduced False Positives:** Minimized impact on legitimate customers

**✅ Real-time Capability:** Model suitable for real-time transaction monitoring

**✅ Actionable Insights:** Identified high-risk patterns and categories

## Risk Mitigation Strategies

**High-Risk Merchant Monitoring:** Enhanced scrutiny for suspicious categories

**Amount Thresholds:** Dynamic limits based on transaction patterns

**Geographic Analysis:** Regional fraud pattern identification

**Device Verification:** Enhanced authentication for suspicious devices

## 🚀 Installation & Usage

**Prerequisites**

pip install -r requirements.txt

## Requirements

pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
Running the Analysis

# Clone repository

git clone https://github.com/yourusername/credit-card-fraud-detection.git

# Navigate to project directory

cd credit-card-fraud-detection

# Run Jupyter notebook

jupyter notebook notebooks/fraud_detection_analysis.ipynb
Model Training Script

python src/model_training.py
🔮 Future Work
Implement Deep Learning models (LSTM, Autoencoders)

Add Real-time streaming capabilities

Develop Ensemble methods for improved performance

Create API deployment for production use

Implement Model monitoring and performance tracking

Add Explainable AI for fraud reason codes

Develop Adaptive learning for evolving fraud patterns

## 👥 Contributors

Bhuvaneshwari - Project Developer

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

Dataset sourced from GitHub Public Repository

Inspired by financial security research literature

Built with scikit-learn and pandas ecosystem

SMOTE implementation from imbalanced-learn library
