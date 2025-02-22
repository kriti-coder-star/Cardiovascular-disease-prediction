# Cardiovascular-disease-prediction
"A machine learning model for predicting cardiovascular disease based on medical data. Utilizes feature engineering, classification algorithms, and data visualization to enhance prediction accuracy."

Overview
This project aims to predict cardiovascular disease using machine learning models. The dataset contains patient health metrics such as age, cholesterol levels, blood pressure, and other vital parameters. Three machine learning models are implemented and compared:
Random Forest Classifier
Logistic Regression
Support Vector Machine (SVM)

Dataset
The dataset used for this project is cardio_train.csv, which consists of patient health records. The target variable cardio indicates the presence (1) or absence (0) of cardiovascular disease.

Features in the Dataset:
Age (in days)
Gender (1 = Male, 2 = Female)
Height (in cm)
Weight (in kg)
Systolic Blood Pressure
Diastolic Blood Pressure
Cholesterol Level (1 = normal, 2 = above normal, 3 = well above normal)
Glucose Level (1 = normal, 2 = above normal, 3 = well above normal)
Smoking Status (0 = No, 1 = Yes)
Alcohol Intake (0 = No, 1 = Yes)
Physical Activity (0 = No, 1 = Yes)
Cardiovascular Disease (Target Variable) (0 = No, 1 = Yes)
Dependencies

Before running the project, ensure you have installed the required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn

Steps to Run the Project
1. Load and Explore the Dataset
Read cardio_train.csv and display basic statistics.
Handle missing values and remove unnecessary columns.

2. Data Preprocessing
Convert categorical variables into numerical form if necessary.
Scale numerical features using StandardScaler.
Split data into training (80%) and testing (20%) sets.

3. Model Training and Evaluation
Random Forest Classifier
Uses n_estimators=100
Evaluates using Accuracy, ROC-AUC, Confusion Matrix, and Feature Importance.
Logistic Regression
Evaluates using Accuracy, ROC-AUC, and Confusion Matrix.
Support Vector Machine (SVM)
Uses LinearSVC for faster training.
Evaluates using Accuracy, ROC-AUC, and Confusion Matrix.

4. Visualization
Confusion matrices for each model.
Feature importance plot for Random Forest.
ROC curves for model comparison.
Running the Script
To run the script and execute the model training, use the following command:
python script.py

Results
The following observations were made from the model evaluations:
Random Forest achieved the highest accuracy and provided insights into feature importance.
Logistic Regression performed well with a balanced approach but had slightly lower accuracy.
SVM showed promising results but required hyperparameter tuning for optimal performance.

Future Enhancements
Perform hyperparameter tuning to improve model performance.
Experiment with additional ML models such as XGBoost and Neural Networks.
Implement better feature engineering techniques to improve prediction accuracy.
Deploy the model as a web application for real-time predictions.

Author
[Kriti Aggarwal]
