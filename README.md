🔮 Customer Churn Prediction
Can we predict who's about to leave before they actually do? This project says: Yes! 🚀
We built a Customer Churn Prediction pipeline that dives into data, battles imbalance 🥊, and optimizes metrics that matter most — precision 🎯 & recall 🔍.

Big thanks to Uneeq Interns 💼 for the opportunity to bring this to life!

📊 Project Overview
From raw CSVs 📂 to actionable insights 📈 — this notebook takes you through:

🧹 Data cleaning & preprocessing

🕵️ Exploratory Data Analysis (EDA)

⚖️ Handling imbalanced classes with SMOTE & undersampling

🤖 Training multiple models (LogisticRegression, RandomForestClassifier, XGBClassifier)

🧮 Evaluating with precision, recall, F1-score, ROC AUC

📂 Dataset
/kaggle/input/customer-churn-train-test/customer_churn_dataset-training-master.csv

/kaggle/input/customer-churn-train-test/customer_churn_dataset-testing-master.csv

🛠️ Tech Stack
Languages & Libraries:

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (classification_report, confusion_matrix,

from sklearn.metrics import auc

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import shap

import warnings

import xgboost as xgb

🧠 Models Trained
LogisticRegression 🤖

RandomForestClassifier 🤖

XGBClassifier 🤖

📈 Results
See the notebook for:

Confusion Matrix 🌀

Classification Report 📜

ROC Curve 📊

Precision & Recall scores 📌

🙏 Acknowledgements
Special thanks to Uneeq Interns 💼 for the mentorship and guidance!
