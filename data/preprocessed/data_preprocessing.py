import pandas as pd     
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

raw_data = pd.read_csv(r"c:\Users\11kun\OneDrive\Documents\Loan Approval System\data\raw\loan-approval-data.csv")

#Handling Missing Data
categorical_cols = raw_data.select_dtypes(include = ["object"]).columns
numerical_cols = raw_data.select_dtypes(include = ["float64", "int64"]).columns

num_imp = SimpleImputer(strategy="mean")
raw_data[numerical_cols] = num_imp.fit_transform(raw_data[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
raw_data[categorical_cols] = cat_imp.fit_transform(raw_data[categorical_cols])

#Removing Unrequired Feature
raw_data = raw_data.drop("Applicant_ID", axis=1)

#Feature Encoding
le = LabelEncoder()
raw_data["Education_Level"] = le.fit_transform(raw_data["Education_Level"])
raw_data["Loan_Approved"] = le.fit_transform(raw_data["Loan_Approved"])

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

encoded = ohe.fit_transform(raw_data[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=raw_data.index)

raw_data = pd.concat([raw_data.drop(columns=cols), encoded_df], axis=1)

#Exporting Preprocessed Data
raw_data.to_csv(r"c:\Users\11kun\OneDrive\Documents\Loan Approval System\data\preprocessed\cleaned_data.csv", index=False)

print(raw_data.head())
print(raw_data.info())
