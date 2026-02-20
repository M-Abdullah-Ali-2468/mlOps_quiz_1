# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/raw/titanic.csv")

# Drop unnecessary columns if exist
if "Name" in data.columns:
    data = data.drop(columns=["Name"])

# Handle missing values
data = data.fillna(0)

# Convert categorical columns to numeric
data = pd.get_dummies(data)

# Separate features and target
if "Survived" in data.columns:
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
else:
    # If dataset does not have Survived column
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data
X_train.to_csv("data/preprocessed/X_train.csv", index=False)
X_test.to_csv("data/preprocessed/X_test.csv", index=False)
y_train.to_csv("data/preprocessed/y_train.csv", index=False)
y_test.to_csv("data/preprocessed/y_test.csv", index=False)

print("Data preprocessing completed")