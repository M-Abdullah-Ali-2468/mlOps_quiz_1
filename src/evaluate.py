# Import libraries
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("models/logistic_model.pkl")

# Load test data
X_test = pd.read_csv("data/preprocessed/X_test.csv")
y_test = pd.read_csv("data/preprocessed/y_test.csv")

y_test = y_test.values.ravel()

# Predict
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save results
with open("results/accuracy.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy))

print("Evaluation completed")
print("Final Accuracy:", accuracy)