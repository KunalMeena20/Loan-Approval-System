import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Importing Cleaned Data
cleaned_data = pd.read_csv(r"c:\Users\11kun\OneDrive\Documents\Loan Approval System\data\preprocessed\cleaned_data.csv")

#Train-Test Split
X = cleaned_data.drop("Loan_Approved", axis=1)
y = cleaned_data["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model Training & Testing (Naive Bayes)
nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)
y_pred = nb_model.predict(X_test_scaled)

#Evaluation
print("Naive Bayes Model")
print("Precision: ", precision_score(y_test, y_pred) )
print("Recall: ", recall_score(y_test, y_pred) )
print("F1 score: ", f1_score(y_test, y_pred) )
print("Accuracy: ", accuracy_score(y_test, y_pred) )
print("CM: ", confusion_matrix(y_test, y_pred) )