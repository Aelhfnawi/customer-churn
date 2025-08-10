Customer Churn Prediction
This repository contains a machine learning project focused on predicting customer churn using a comprehensive dataset. The goal is to build a robust model that can identify customers at risk of churning, allowing businesses to implement proactive retention strategies.

Dataset
The project uses the customer_churn_dataset-training-master.csv dataset, which contains various features related to customer behavior and demographics. The key features include:

CustomerID: Unique identifier for each customer.

Age: Customer's age.

Gender: Customer's gender.

Tenure: Duration of the customer's subscription.

Usage Frequency: How often the customer uses the service.

Support Calls: Number of times the customer has contacted support.

Payment Delay: The average delay in days for payments.

Subscription Type: The type of subscription (e.g., Standard, Basic).

Contract Length: The duration of the customer's contract (e.g., Monthly, Annual).

Total Spend: The total amount the customer has spent.

Last Interaction: Days since the last interaction with the service.

Churn: The target variable, indicating whether the customer has churned (1) or not (0).

Methodology
The Jupyter notebook customer-churn.ipynb follows a standard machine learning pipeline:

Data Loading and Exploration: The process begins with loading the training and testing datasets and performing an initial analysis to understand the data's shape, column types, missing values, and the distribution of the target variable.

Data Preprocessing: This step handles data cleaning and feature engineering. It includes:

Missing Value Handling: Rows with missing values are dropped.

Categorical Encoding: Categorical features like Gender, Subscription Type, and Contract Length are converted into numerical representations using LabelEncoder.

Feature Scaling: Numerical features are scaled using StandardScaler to ensure they contribute equally to the model.

Model Training: The notebook uses a variety of machine learning models to predict churn. The models are trained and evaluated on the preprocessed data.

Model Evaluation: The performance of the models is evaluated using a range of metrics, including a classification report, confusion matrix, precision-recall curve, F1 score, and ROC AUC score.

Hyperparameter Tuning: To optimize model performance, hyperparameter tuning is performed using GridSearchCV.

Feature Importance Analysis: The project uses SHAP (SHapley Additive exPlanations) to understand which features are most influential in the model's predictions. This provides valuable insights into the key drivers of customer churn.

Project Structure
customer-churn.ipynb: The main Jupyter notebook containing the full analysis and model pipeline.

customer_churn_dataset-training-master.csv: The training dataset used in this project.

customer_churn_dataset-testing-master.csv: The testing dataset used to evaluate the model's performance.

Requirements
The following Python libraries are required to run the notebook:

pandas

numpy

matplotlib

seaborn

scikit-learn

imblearn (for handling imbalanced data)

xgboost

shap
